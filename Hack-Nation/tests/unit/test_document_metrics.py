"""Document metric correctness.

The interesting cases are the ones that distinguish failure modes: a value that
is right in the source unit but wrong after conversion must score differently
from one that was misread outright.
"""

from __future__ import annotations

import pytest

from evaluation.documents import (
    analyte_name_prf,
    date_prf,
    evaluate_documents,
    match_results,
    numbers_match,
    numeric_value_prf,
    page_grounding_accuracy,
    reference_range_prf,
    source_value_prf,
    unit_conversion_rate,
    unit_prf,
    unsupported_value_rate,
)


def value(code, **kwargs):
    base = {
        "document_id": "DOC-1",
        "canonical_code": code,
        "value_source": 2.7,
        "unit_source": "nmol/L",
        "value_canonical": 77.8086,
        "unit_canonical": "ng/dL",
        "conversion_applied": True,
        "reference_low": 0.5,
        "reference_high": 2.4,
        "collected_date": "2024-03-14",
        "page": 1,
    }
    base.update(kwargs)
    return base


# -- Numeric comparison -----------------------------------------------------


def test_numbers_match_within_tolerance():
    assert numbers_match(78.0, 78) is True
    assert numbers_match(1.0, 1.0000000001) is True


def test_numbers_match_rejects_none():
    assert numbers_match(None, 1.0) is False
    assert numbers_match(1.0, None) is False
    assert numbers_match(None, None) is False


def test_numbers_match_rejects_real_differences():
    assert numbers_match(78.0, 79.0) is False


# -- Matching ---------------------------------------------------------------


def test_matching_is_per_document_and_code():
    gold = [value("total_testosterone"), value("shbg", document_id="DOC-2")]
    pred = [value("shbg", document_id="DOC-2"), value("total_testosterone")]
    matched, spurious, missed = match_results(gold, pred)
    assert len(matched) == 2 and not spurious and not missed


def test_same_code_in_a_different_document_is_not_a_match():
    gold = [value("total_testosterone", document_id="DOC-1")]
    pred = [value("total_testosterone", document_id="DOC-2")]
    matched, spurious, missed = match_results(gold, pred)
    assert not matched and len(spurious) == 1 and len(missed) == 1


def test_analyte_name_prf():
    gold = [value("total_testosterone"), value("shbg")]
    pred = [value("total_testosterone"), value("dheas")]
    score = analyte_name_prf(gold, pred)
    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1


# -- Value and unit metrics -------------------------------------------------


def test_correct_conversion_scores_full_marks():
    gold = [value("total_testosterone")]
    pred = [value("total_testosterone")]
    assert numeric_value_prf(gold, pred).f1 == 1.0
    assert source_value_prf(gold, pred).f1 == 1.0
    assert unit_prf(gold, pred).f1 == 1.0


def test_a_conversion_bug_is_visible_in_the_canonical_metric_only():
    """Source value right, canonical value wrong: exactly a unit-factor bug."""
    gold = [value("total_testosterone")]
    pred = [value("total_testosterone", value_canonical=2.7, unit_canonical="nmol/L")]
    assert source_value_prf(gold, pred).f1 == 1.0
    assert numeric_value_prf(gold, pred).f1 == 0.0
    assert unit_prf(gold, pred).f1 == 0.0


def test_a_misread_number_is_visible_in_both_value_metrics():
    gold = [value("total_testosterone")]
    pred = [value("total_testosterone", value_source=27.0, value_canonical=778.086)]
    assert source_value_prf(gold, pred).f1 == 0.0
    assert numeric_value_prf(gold, pred).f1 == 0.0


def test_a_missing_canonical_value_never_counts_as_correct():
    gold = [value("total_testosterone")]
    pred = [value("total_testosterone", value_canonical=None)]
    assert numeric_value_prf(gold, pred).f1 == 0.0


# -- Dates and reference ranges --------------------------------------------


def test_date_prf():
    gold = [value("total_testosterone")]
    pred = [value("total_testosterone", collected_date="2024-03-15")]
    assert date_prf(gold, pred).f1 == 0.0
    assert date_prf(gold, gold).f1 == 1.0


def test_reference_range_prf_scores_only_annotated_ranges():
    gold = [value("total_testosterone", reference_low=None, reference_high=None)]
    pred = [value("total_testosterone", reference_low=None, reference_high=None)]
    assert reference_range_prf(gold, pred).support == 0


def test_reference_range_prf_detects_a_wrong_bound():
    gold = [value("total_testosterone")]
    pred = [value("total_testosterone", reference_high=24.0)]
    assert reference_range_prf(gold, pred).f1 == 0.0


def test_one_sided_reference_range_matches():
    gold = [value("triglycerides", reference_low=None, reference_high=1.7)]
    pred = [value("triglycerides", reference_low=None, reference_high=1.7)]
    assert reference_range_prf(gold, pred).f1 == 1.0


# -- Grounding and rates ----------------------------------------------------


def test_page_grounding_accuracy():
    gold = [value("total_testosterone", page=1), value("shbg", page=2)]
    pred = [value("total_testosterone", page=1), value("shbg", page=1)]
    assert page_grounding_accuracy(gold, pred) == pytest.approx(0.5)


def test_page_grounding_with_no_matches_is_zero():
    assert page_grounding_accuracy([], []) == 0.0


def test_unsupported_value_rate():
    assert unsupported_value_rate(9, 1) == pytest.approx(0.1)
    assert unsupported_value_rate(0, 0) == 0.0


def test_unit_conversion_rate_shows_the_path_was_exercised():
    predicted = [
        value("total_testosterone", conversion_applied=True),
        value("shbg", conversion_applied=False),
    ]
    assert unit_conversion_rate(predicted) == pytest.approx(0.5)
    assert unit_conversion_rate([]) == 0.0


# -- Report assembly --------------------------------------------------------


def test_evaluate_documents_assembles_a_serializable_report():
    gold = [value("total_testosterone"), value("shbg", value_source=28.0)]
    pred = [value("total_testosterone"), value("shbg", value_source=28.0)]
    report = evaluate_documents(
        gold, pred, corpus_id="test_docs", n_documents=1, n_unsupported=1, n_unmapped=2
    )
    assert report.test_name.f1 == 1.0
    assert report.canonical_value.f1 == 1.0
    assert report.page_grounding_accuracy == 1.0
    assert report.n_unmapped_tests == 2
    assert report.unsupported_value_rate == pytest.approx(1 / 3, abs=1e-4)

    import json

    json.dumps(report.model_dump(mode="json"))  # must not raise
