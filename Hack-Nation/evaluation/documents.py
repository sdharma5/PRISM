"""Metrics for the document-extraction pipeline.

Why field-level metrics rather than one accuracy number: the four things an
extractor must get right — which test, what number, what unit, which page — fail
independently and have different consequences. Reading the right number under
the wrong test name produces a confidently wrong dataset. Reading the right
number with the wrong unit produces an off-by-28 testosterone. A single blended
score would let a pipeline with one catastrophic failure mode look "95%
accurate", so each field is scored on its own and reported side by side.

Numeric comparison uses a relative tolerance rather than equality because a
report printing 78.0 and an extractor yielding 78 are the same measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_RTOL = 1e-6


class PrfScore(BaseModel):
    """Precision / recall / F1 with the counts that produced them."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    support: int = 0


def _prf(tp: int, fp: int, fn: int) -> PrfScore:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PrfScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        support=tp + fn,
    )


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)


def numbers_match(a: Any, b: Any, *, rtol: float = DEFAULT_RTOL) -> bool:
    """Relative-tolerance comparison that treats ``None`` as unequal to anything."""
    if a is None or b is None:
        return False
    try:
        left, right = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if left == right:
        return True
    scale = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / scale <= rtol


def match_results(
    gold: Sequence[Any], predicted: Sequence[Any]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Align predictions to gold on ``(document_id, canonical_code)``.

    A document may legitimately report the same analyte twice (two draw dates),
    so matching is greedy over the remaining candidates rather than a dict merge.
    """
    gold_items = [_as_dict(g) for g in gold]
    pred_items = [_as_dict(p) for p in predicted]
    remaining = list(pred_items)
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missed: list[dict[str, Any]] = []

    def key(item: dict[str, Any]) -> tuple[str, str]:
        return (str(item.get("document_id", "")), str(item.get("canonical_code", "")))

    for gold_item in gold_items:
        candidate = next((p for p in remaining if key(p) == key(gold_item)), None)
        if candidate is None:
            missed.append(gold_item)
        else:
            remaining.remove(candidate)
            matched.append((gold_item, candidate))
    return matched, remaining, missed


def analyte_name_prf(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    """Did we find the right analytes at all?"""
    matched, spurious, missed = match_results(gold, predicted)
    return _prf(len(matched), len(spurious), len(missed))


def numeric_value_prf(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    *,
    field: str = "value_canonical",
    rtol: float = 1e-3,
) -> PrfScore:
    """Correctness of the canonical numeric value on matched analytes."""
    matched, spurious, missed = match_results(gold, predicted)
    tp = sum(1 for g, p in matched if numbers_match(g.get(field), p.get(field), rtol=rtol))
    fp = (len(matched) - tp) + len(spurious)
    return _prf(tp, fp, len(missed))


def source_value_prf(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    """Correctness of the value exactly as printed in the report.

    Scored separately from the canonical value so that a unit-conversion bug and
    a number-parsing bug cannot be mistaken for each other.
    """
    return numeric_value_prf(gold, predicted, field="value_source", rtol=1e-9)


def unit_prf(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    """Correctness of the canonical unit on matched analytes."""
    matched, spurious, missed = match_results(gold, predicted)
    tp = sum(
        1
        for g, p in matched
        if str(g.get("unit_canonical") or "") == str(p.get("unit_canonical") or "")
    )
    fp = (len(matched) - tp) + len(spurious)
    return _prf(tp, fp, len(missed))


def date_prf(
    gold: Sequence[Any], predicted: Sequence[Any], *, field: str = "collected_date"
) -> PrfScore:
    """Correctness of an extracted date field on matched analytes."""
    matched, spurious, missed = match_results(gold, predicted)
    tp = sum(1 for g, p in matched if str(g.get(field) or "") == str(p.get(field) or ""))
    fp = (len(matched) - tp) + len(spurious)
    return _prf(tp, fp, len(missed))


def reference_range_prf(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    """Correctness of the parsed reference bounds on matched analytes.

    Only scored where the gold annotation states a range; an analyte the report
    printed without one is not a miss.
    """
    matched, spurious, missed = match_results(gold, predicted)
    tp = fp = fn = 0
    for gold_item, pred_item in matched:
        gold_low, gold_high = gold_item.get("reference_low"), gold_item.get("reference_high")
        if gold_low is None and gold_high is None:
            continue
        pred_low, pred_high = pred_item.get("reference_low"), pred_item.get("reference_high")
        low_ok = (gold_low is None and pred_low is None) or numbers_match(
            gold_low, pred_low, rtol=1e-6
        )
        high_ok = (gold_high is None and pred_high is None) or numbers_match(
            gold_high, pred_high, rtol=1e-6
        )
        if low_ok and high_ok:
            tp += 1
        else:
            fp += 1
    fn += sum(
        1
        for g in missed
        if g.get("reference_low") is not None or g.get("reference_high") is not None
    )
    fp += sum(
        1
        for p in spurious
        if p.get("reference_low") is not None or p.get("reference_high") is not None
    )
    return _prf(tp, fp, fn)


def page_grounding_accuracy(gold: Sequence[Any], predicted: Sequence[Any]) -> float:
    """Share of matched values placed on the correct page."""
    matched, _spurious, _missed = match_results(gold, predicted)
    if not matched:
        return 0.0
    correct = sum(1 for g, p in matched if int(g.get("page", 0)) == int(p.get("page", -1)))
    return round(correct / len(matched), 4)


def unsupported_value_rate(n_extracted: int, n_unsupported: int) -> float:
    """Share of candidate values that were dropped for lacking a page span."""
    total = n_extracted + n_unsupported
    if total == 0:
        return 0.0
    return round(n_unsupported / total, 4)


def unit_conversion_rate(predicted: Sequence[Any]) -> float:
    """Share of extracted values that actually required a unit conversion.

    Reported so a reviewer can see the conversion path was genuinely exercised
    rather than every value already arriving in canonical units.
    """
    items = [_as_dict(p) for p in predicted]
    if not items:
        return 0.0
    converted = sum(1 for p in items if p.get("conversion_applied"))
    return round(converted / len(items), 4)


class DocumentEvaluationReport(BaseModel):
    """Everything one document-evaluation run produced."""

    corpus_id: str
    n_documents: int = 0
    n_gold_values: int = 0
    n_extracted_values: int = 0

    test_name: PrfScore = Field(default_factory=PrfScore)
    source_value: PrfScore = Field(default_factory=PrfScore)
    canonical_value: PrfScore = Field(default_factory=PrfScore)
    unit: PrfScore = Field(default_factory=PrfScore)
    collected_date: PrfScore = Field(default_factory=PrfScore)
    reference_range: PrfScore = Field(default_factory=PrfScore)

    page_grounding_accuracy: float = 0.0
    unsupported_value_rate: float = 0.0
    unit_conversion_rate: float = 0.0

    n_unmapped_tests: int = 0
    notes: list[str] = Field(default_factory=list)


def evaluate_documents(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    *,
    corpus_id: str,
    n_documents: int,
    n_unsupported: int = 0,
    n_unmapped: int = 0,
) -> DocumentEvaluationReport:
    """Compute the full document metric set for one corpus run."""
    return DocumentEvaluationReport(
        corpus_id=corpus_id,
        n_documents=n_documents,
        n_gold_values=len(gold),
        n_extracted_values=len(predicted),
        test_name=analyte_name_prf(gold, predicted),
        source_value=source_value_prf(gold, predicted),
        canonical_value=numeric_value_prf(gold, predicted),
        unit=unit_prf(gold, predicted),
        collected_date=date_prf(gold, predicted),
        reference_range=reference_range_prf(gold, predicted),
        page_grounding_accuracy=page_grounding_accuracy(gold, predicted),
        unsupported_value_rate=unsupported_value_rate(len(predicted), n_unsupported),
        unit_conversion_rate=unit_conversion_rate(predicted),
        n_unmapped_tests=n_unmapped,
    )
