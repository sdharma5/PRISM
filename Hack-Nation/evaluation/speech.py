"""Metrics for the speech pipeline.

Why implement Levenshtein here: the only optional dependency that would provide
it (``jiwer``) is not in the test path, and a WER implementation is fifteen lines
of dynamic programming. The optional import is attempted first purely so that a
user who has jiwer gets its (identical) numbers with its tokenization.

Why so many attribute-level metrics: an extractor that finds "acne" in every
sentence but gets negation wrong is *worse* than one that finds nothing, because
it manufactures false findings that a reviewer must catch. F1 on concept
detection alone would rate them the same, so negation, temporality and
attribution are scored separately and reported alongside.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field


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


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def normalize_for_wer(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace into tokens."""
    lowered = text.lower().replace("'", "").replace("’", "")
    return [t for t in re.split(r"[^a-z0-9]+", lowered) if t]


def levenshtein(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Edit distance between two token sequences (substitution cost 1)."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_token in enumerate(hypothesis, start=1):
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + (ref_token != hyp_token),  # substitution
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, *, use_jiwer: bool = True) -> float:
    """WER = edit distance / reference length. Empty reference yields 0.0 or 1.0."""
    if use_jiwer:
        try:  # pragma: no cover - optional dependency
            import jiwer

            return float(jiwer.wer(reference, hypothesis))
        except ImportError:
            pass
    ref_tokens = normalize_for_wer(reference)
    hyp_tokens = normalize_for_wer(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return round(levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens), 4)


def corpus_word_error_rate(pairs: Iterable[tuple[str, str]]) -> float:
    """Length-weighted WER across a corpus of (reference, hypothesis) pairs."""
    total_errors = 0
    total_words = 0
    for reference, hypothesis in pairs:
        ref_tokens = normalize_for_wer(reference)
        total_errors += levenshtein(ref_tokens, normalize_for_wer(hypothesis))
        total_words += len(ref_tokens)
    if total_words == 0:
        return 0.0
    return round(total_errors / total_words, 4)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)


def _key(item: dict[str, Any]) -> tuple[str, str]:
    """Identity of an assertion for matching: what, about whom."""
    return (
        str(item.get("canonical_code") or item.get("code") or ""),
        str(item.get("attribution", "patient")),
    )


def match_events(
    gold: Sequence[Any], predicted: Sequence[Any]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedily align predictions to gold by ``(code, attribution)``.

    Attribute correctness (negation, temporality, speaker) is scored only on
    matched pairs, because scoring negation on an event that was never found
    would conflate two different failures.
    """
    gold_items = [_as_dict(g) for g in gold]
    pred_items = [_as_dict(p) for p in predicted]
    unmatched_pred = list(pred_items)
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missed: list[dict[str, Any]] = []

    for gold_item in gold_items:
        candidate = next((p for p in unmatched_pred if _key(p) == _key(gold_item)), None)
        if candidate is None:
            missed.append(gold_item)
        else:
            unmatched_pred.remove(candidate)
            matched.append((gold_item, candidate))

    return matched, unmatched_pred, missed


def symptom_extraction_prf(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    """Concept-level precision / recall / F1."""
    matched, spurious, missed = match_events(gold, predicted)
    return _prf(len(matched), len(spurious), len(missed))


def _attribute_prf(
    matched: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    missed: Sequence[dict[str, Any]],
    spurious: Sequence[dict[str, Any]],
    attribute: str,
) -> PrfScore:
    """Binary F1 for a boolean attribute, charging unmatched events too.

    A gold negated event that was never extracted is a false negative for
    negation as well as for detection: the reviewer still never saw it.
    """
    tp = fp = fn = 0
    for gold_item, pred_item in matched:
        gold_flag = bool(gold_item.get(attribute, False))
        pred_flag = bool(pred_item.get(attribute, False))
        if gold_flag and pred_flag:
            tp += 1
        elif pred_flag and not gold_flag:
            fp += 1
        elif gold_flag and not pred_flag:
            fn += 1
    fn += sum(1 for g in missed if bool(g.get(attribute, False)))
    fp += sum(1 for p in spurious if bool(p.get(attribute, False)))
    return _prf(tp, fp, fn)


def negation_f1(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    matched, spurious, missed = match_events(gold, predicted)
    return _attribute_prf(matched, missed, spurious, "negated")


def temporality_f1(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    """F1 on the ``historical`` flag."""
    matched, spurious, missed = match_events(gold, predicted)
    return _attribute_prf(matched, missed, spurious, "historical")


def uncertainty_f1(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    matched, spurious, missed = match_events(gold, predicted)
    return _attribute_prf(matched, missed, spurious, "uncertain")


def medication_event_f1(gold: Sequence[Any], predicted: Sequence[Any]) -> PrfScore:
    """F1 over (medication name, action) pairs.

    Matched on the drug and what happened to it, because "started metformin" and
    "stopped metformin" are opposite clinical facts sharing a code.
    """

    def med_keys(items: Sequence[Any]) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        for raw in items:
            item = _as_dict(raw)
            code = str(item.get("canonical_code") or item.get("code") or "")
            if code != "medication_current":
                continue
            keys.append((str(item.get("value", "")), str(item.get("medication_action") or "")))
        return keys

    gold_keys = med_keys(gold)
    pred_keys = med_keys(predicted)
    remaining = list(pred_keys)
    tp = 0
    for key in gold_keys:
        if key in remaining:
            remaining.remove(key)
            tp += 1
    return _prf(tp, len(remaining), len(gold_keys) - tp)


def speaker_attribution_accuracy(gold: Sequence[Any], predicted: Sequence[Any]) -> float:
    """Fraction of matched events whose speaker role was identified correctly."""
    matched, _spurious, _missed = match_events(gold, predicted)
    if not matched:
        return 0.0
    correct = sum(
        1
        for gold_item, pred_item in matched
        if str(gold_item.get("speaker_role", "patient")) == str(pred_item.get("speaker_role", ""))
    )
    return round(correct / len(matched), 4)


def unsupported_event_rate(predicted: Sequence[Any], n_unsupported: int) -> float:
    """Share of predictions that carried no verifiable supporting span."""
    total = len(predicted) + n_unsupported
    if total == 0:
        return 0.0
    return round(n_unsupported / total, 4)


def user_correction_rate(n_reviewed: int, n_corrected: int) -> float:
    """Share of reviewed extractions a human had to change."""
    if n_reviewed <= 0:
        return 0.0
    return round(n_corrected / n_reviewed, 4)


class SpeechEvaluationReport(BaseModel):
    """Everything one evaluation run produced, ready to serialize to JSON."""

    corpus_id: str
    n_utterances: int = 0
    n_gold_events: int = 0
    n_predicted_events: int = 0

    word_error_rate: float = 0.0
    symptom_extraction: PrfScore = Field(default_factory=PrfScore)
    negation: PrfScore = Field(default_factory=PrfScore)
    temporality: PrfScore = Field(default_factory=PrfScore)
    uncertainty: PrfScore = Field(default_factory=PrfScore)
    medication_events: PrfScore = Field(default_factory=PrfScore)

    speaker_attribution_accuracy: float = 0.0
    unsupported_event_rate: float = 0.0
    user_correction_rate: float = 0.0

    per_category: dict[str, PrfScore] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def evaluate_corpus(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    *,
    corpus_id: str,
    n_utterances: int,
    transcript_pairs: Sequence[tuple[str, str]] = (),
    n_unsupported: int = 0,
    n_reviewed: int = 0,
    n_corrected: int = 0,
    categories: dict[str, tuple[Sequence[Any], Sequence[Any]]] | None = None,
) -> SpeechEvaluationReport:
    """Compute the full metric set for one corpus run."""
    report = SpeechEvaluationReport(
        corpus_id=corpus_id,
        n_utterances=n_utterances,
        n_gold_events=len(gold),
        n_predicted_events=len(predicted),
        word_error_rate=corpus_word_error_rate(transcript_pairs),
        symptom_extraction=symptom_extraction_prf(gold, predicted),
        negation=negation_f1(gold, predicted),
        temporality=temporality_f1(gold, predicted),
        uncertainty=uncertainty_f1(gold, predicted),
        medication_events=medication_event_f1(gold, predicted),
        speaker_attribution_accuracy=speaker_attribution_accuracy(gold, predicted),
        unsupported_event_rate=unsupported_event_rate(predicted, n_unsupported),
        user_correction_rate=user_correction_rate(n_reviewed, n_corrected),
    )
    for name, (cat_gold, cat_pred) in (categories or {}).items():
        report.per_category[name] = symptom_extraction_prf(cat_gold, cat_pred)
    return report
