"""Grounding: an ungrounded value must be dropped and counted, never stored.

This is the document-side equivalent of the confirmation boundary. An extractor
that emits a number nobody can point at in the source is indistinguishable from
one that invents numbers, so the pipeline is required to notice and say so.
"""

from __future__ import annotations

import pytest

from evaluation.documents import unsupported_value_rate
from ingestion.documents.lab_extractor import ExtractedLabResult, LabExtractor
from ingestion.documents.parser import TextFixtureParser
from ingestion.documents.validation import (
    drop_ungrounded,
    is_grounded,
    validate_lab_results,
)

REPORT = """[PAGE 1]
SYNTHETIC LABORATORY REPORT - NOT A REAL PATIENT
Collected: 2024-03-14

Total Testosterone: 78 ng/dL (Ref: 15 - 70)
SHBG: 28 nmol/L (Ref: 18 - 114)
"""


@pytest.fixture(scope="module")
def parser() -> TextFixtureParser:
    return TextFixtureParser()


@pytest.fixture(scope="module")
def document(parser):
    return parser.parse(REPORT, document_id="DOC-GROUND")


@pytest.fixture(scope="module")
def results(document):
    return LabExtractor().extract(document, patient_id="P-GROUND").results


def fabricate(template: ExtractedLabResult, **overrides) -> ExtractedLabResult:
    """Copy without re-validating — used to simulate a buggy extractor's output."""
    return template.model_copy(update=overrides)


def rebuild(template: ExtractedLabResult, **overrides) -> ExtractedLabResult:
    """Re-construct through the validators, so model invariants actually fire."""
    return ExtractedLabResult.model_validate({**template.model_dump(), **overrides})


# -- The happy path ---------------------------------------------------------


def test_extracted_values_are_grounded(document, results):
    assert results
    for result in results:
        assert is_grounded(document, result) is True


def test_grounding_survives_the_validation_report(document, results):
    report = validate_lab_results(document, results)
    assert report.ungrounded_result_ids == []
    assert report.grounding_accuracy == 1.0
    assert report.ok is True


# -- Fabricated / drifted spans --------------------------------------------


def test_a_span_pointing_at_the_wrong_text_is_ungrounded(document, results):
    drifted = fabricate(results[0], char_start=0, char_end=8)
    assert is_grounded(document, drifted) is False


def test_a_value_absent_from_its_own_evidence_is_ungrounded(document, results):
    """The line exists, but it does not contain the number that was reported."""
    fabricated = fabricate(results[0], value_source=999.0)
    assert is_grounded(document, fabricated) is False


def test_a_span_on_a_nonexistent_page_is_ungrounded(document, results):
    off_page = fabricate(results[0], page_number=99)
    assert is_grounded(document, off_page) is False


def test_a_span_past_the_end_of_the_document_is_ungrounded(document, results):
    overflow = fabricate(
        results[0], char_start=len(document.text) + 10, char_end=len(document.text) + 50
    )
    assert is_grounded(document, overflow) is False


# -- Dropping and counting --------------------------------------------------


def test_ungrounded_values_are_dropped_and_counted(document, results):
    fabricated = fabricate(results[0], value_source=999.0)
    grounded, ungrounded = drop_ungrounded(document, [*results, fabricated])

    assert fabricated not in grounded
    assert ungrounded == [fabricated]
    assert len(grounded) == len(results)


def test_dropped_values_raise_a_validation_error(document, results):
    fabricated = fabricate(results[0], value_source=999.0)
    report = validate_lab_results(document, [*results, fabricated])

    assert fabricated.result_id in report.ungrounded_result_ids
    assert report.ok is False
    assert report.grounding_accuracy < 1.0
    assert any("dropped" in issue.message for issue in report.issues)


def test_unsupported_rate_reflects_the_drop(document, results):
    fabricated = fabricate(results[0], value_source=999.0)
    grounded, ungrounded = drop_ungrounded(document, [*results, fabricated])
    rate = unsupported_value_rate(len(grounded), len(ungrounded))
    assert rate == pytest.approx(1 / (len(results) + 1), abs=1e-4)


def test_extraction_result_reports_its_own_unsupported_rate(document):
    result = LabExtractor().extract(document, patient_id="P-GROUND")
    assert result.unsupported_rate == 0.0
    assert result.results


# -- The model refuses to hold an ungrounded value --------------------------


def test_an_empty_span_cannot_be_constructed(results):
    with pytest.raises(ValueError, match="non-empty"):
        rebuild(results[0], char_start=10, char_end=10)


def test_blank_evidence_text_cannot_be_constructed(results):
    with pytest.raises(ValueError, match="evidence_text"):
        rebuild(results[0], evidence_text="   ")


def test_page_numbers_are_one_based(results):
    with pytest.raises(ValueError, match="1-based"):
        rebuild(results[0], page_number=0)


# -- Out-of-range values are flagged, not corrected -------------------------


def test_implausible_value_is_flagged_and_preserved(parser):
    """A decimal point lost in the text layer must be visible, not silently fixed."""
    doc = parser.parse(
        "[PAGE 1]\nCollected: 2024-03-14\nTotal Testosterone: 7800 ng/dL (Ref: 15 - 70)\n",
        document_id="DOC-RANGE",
    )
    result = LabExtractor().extract(doc, patient_id="P")
    extracted = result.results[0]
    assert extracted.value_source == pytest.approx(7800.0)
    assert extracted.value_canonical == pytest.approx(7800.0)

    report = validate_lab_results(doc, result.results)
    assert any("outside the registry valid range" in i.message for i in report.issues)
    # Flagged, not clipped.
    assert extracted.value_canonical == pytest.approx(7800.0)


def test_unconvertible_unit_preserves_the_source_value(parser):
    doc = parser.parse(
        "[PAGE 1]\nCollected: 2024-03-14\nTotal Testosterone: 78 furlongs (Ref: 15 - 70)\n",
        document_id="DOC-BADUNIT",
    )
    result = LabExtractor().extract(doc, patient_id="P")
    extracted = result.results[0]
    assert extracted.value_source == pytest.approx(78.0)
    assert extracted.unit_source == "furlongs"
    assert extracted.value_canonical is None
    assert any("unit_conversion_failed" in w for w in extracted.warnings)
