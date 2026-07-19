"""Document extraction: values, units, reference ranges, dates, page grounding.

The unit tests here deliberately assert on BOTH the source and the canonical
representation of every converted value. A test that only checked the canonical
number would pass even if the pipeline threw away what the lab actually printed.
"""

from __future__ import annotations

import pytest

from ingestion.documents.lab_extractor import LabExtractor, parse_reference_range, to_events
from ingestion.documents.normalization import (
    normalize_test_name,
    normalize_unit_text,
    parse_date,
    parse_number,
)
from ingestion.documents.parser import TextFixtureParser
from ingestion.documents.report_extractor import ReportExtractor, findings_to_events

TABLE_REPORT = """[PAGE 1]
SYNTHETIC ENDOCRINE LABORATORY REPORT - NOT A REAL PATIENT
Patient ID: SYNTH-DOC-TEST
Collected: 2024-03-14
Reported: 2024-03-16

[TABLE]
Test | Result | Units | Reference Range
Total Testosterone | 2.7 | nmol/L | 0.5 - 2.4
SHBG | 28 | nmol/L | 18 - 114
[/TABLE]
[PAGE 2]
Fasting Glucose: 5.6 mmol/L (Ref: 3.9 - 5.5)
Triglycerides: 1.9 mmol/L (Ref: <1.7)
"""

ULTRASOUND_REPORT = """[PAGE 1]
SYNTHETIC PELVIC ULTRASOUND REPORT - NOT A REAL PATIENT
Examination date: 2024-05-02

FINDINGS
Right ovary volume: 12.4 mL
Right ovary: 19 antral follicles
Left ovary volume: 9.8 mL
Left ovary: 14 antral follicles

IMPRESSION: Appearances are consistent with polycystic ovarian morphology.
"""


@pytest.fixture(scope="module")
def parser() -> TextFixtureParser:
    return TextFixtureParser()


@pytest.fixture(scope="module")
def document(parser):
    return parser.parse(TABLE_REPORT, document_id="DOC-TEST")


@pytest.fixture(scope="module")
def results(document):
    return LabExtractor().extract(document, patient_id="P-DOC").results


def by_code(results, code):
    matches = [r for r in results if r.canonical_code == code]
    assert matches, f"expected '{code}', got {sorted(r.canonical_code for r in results)}"
    return matches[0]


# -- Parsing ----------------------------------------------------------------


def test_pages_are_split_on_markers(document):
    assert document.n_pages == 2
    assert [p.page_number for p in document.pages] == [1, 2]


def test_tables_are_structured(document):
    tables = document.tables()
    assert len(tables) == 1
    assert tables[0].header == ["Test", "Result", "Units", "Reference Range"]
    assert tables[0].rows[0][0] == "Total Testosterone"


def test_a_document_without_markers_is_one_page(parser):
    doc = parser.parse("Total Testosterone: 78 ng/dL", document_id="D")
    assert doc.n_pages == 1
    assert doc.pages[0].page_number == 1


# -- Values and units -------------------------------------------------------


def test_table_value_is_extracted(results):
    result = by_code(results, "total_testosterone")
    assert result.value_source == pytest.approx(2.7)
    assert result.unit_source == "nmol/L"
    assert result.extraction_method == "table"


def test_line_value_is_extracted(results):
    result = by_code(results, "fasting_glucose")
    assert result.value_source == pytest.approx(5.6)
    assert result.unit_source == "mmol/L"
    assert result.extraction_method == "line"


def test_unit_conversion_preserves_both_representations(results):
    """The whole point: the lab's number AND the canonical number both survive."""
    result = by_code(results, "total_testosterone")
    assert result.value_source == pytest.approx(2.7)
    assert result.unit_source == "nmol/L"
    assert result.value_canonical == pytest.approx(2.7 * 28.818)
    assert result.unit_canonical == "ng/dL"
    assert result.conversion_applied is True


def test_glucose_is_converted_from_mmol(results):
    result = by_code(results, "fasting_glucose")
    assert result.value_canonical == pytest.approx(5.6 * 18.016)
    assert result.unit_canonical == "mg/dL"
    assert result.conversion_applied is True


def test_already_canonical_units_are_not_flagged_as_converted(results):
    result = by_code(results, "shbg")
    assert result.unit_source == "nmol/L"
    assert result.unit_canonical == "nmol/L"
    assert result.value_canonical == pytest.approx(28.0)
    assert result.conversion_applied is False


def test_events_carry_source_and_canonical_values(results):
    events = to_events(results, source_dataset="prism_document_eval_synthetic")
    event = next(e for e in events if e.canonical_variable_code == "total_testosterone")
    assert event.raw_value == pytest.approx(2.7)
    assert event.raw_unit == "nmol/L"
    assert event.value == pytest.approx(2.7 * 28.818)
    assert event.unit == "ng/dL"
    assert event.provenance == "document_extracted"
    assert event.confirmation_status == "awaiting_clinician_confirmation"
    assert event.is_model_ready is False


# -- Reference ranges -------------------------------------------------------


def test_between_reference_range(results):
    reference = by_code(results, "total_testosterone").reference_range
    assert reference is not None
    assert reference.operator == "between"
    assert reference.low == pytest.approx(0.5)
    assert reference.high == pytest.approx(2.4)
    assert reference.contains(2.7) is False


def test_upper_bound_reference_range(results):
    reference = by_code(results, "triglycerides").reference_range
    assert reference is not None
    assert reference.operator == "lt"
    assert reference.high == pytest.approx(1.7)
    assert reference.contains(1.9) is False


@pytest.mark.parametrize(
    ("text", "operator", "low", "high"),
    [
        ("2.5 - 10.2", "between", 2.5, 10.2),
        ("2.5-10.2", "between", 2.5, 10.2),
        ("2.5 to 10.2", "between", 2.5, 10.2),
        ("Ref: 15 - 70", "between", 15.0, 70.0),
        ("<1.5", "lt", None, 1.5),
        ("> 40", "gt", 40.0, None),
        (">=40", "gte", 40.0, None),
        ("2.5 – 10.2", "between", 2.5, 10.2),  # en dash
    ],
)
def test_reference_range_parsing(text, operator, low, high):
    reference = parse_reference_range(text)
    assert reference is not None
    assert reference.operator == operator
    assert (reference.low is None and low is None) or reference.low == pytest.approx(low)
    assert (reference.high is None and high is None) or reference.high == pytest.approx(high)


def test_unparseable_reference_range_is_none():
    assert parse_reference_range("see comment") is None
    assert parse_reference_range("") is None


# -- Dates ------------------------------------------------------------------


def test_collected_and_reported_dates(results):
    result = by_code(results, "total_testosterone")
    assert result.collected_date == "2024-03-14"
    assert result.reported_date == "2024-03-16"


@pytest.mark.parametrize(
    ("raw", "iso"),
    [
        ("2024-03-14", "2024-03-14"),
        ("14/03/2024", "2024-03-14"),
        ("14 Mar 2024", "2024-03-14"),
        ("March 14, 2024", "2024-03-14"),
        ("14.03.2024", "2024-03-14"),
    ],
)
def test_date_formats(raw, iso):
    parsed = parse_date(raw)
    assert parsed is not None
    assert parsed.isoformat() == iso


def test_unparseable_date_is_none():
    assert parse_date("last Tuesday") is None


# -- Normalization ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234", 1.234),
        ("1,234", 1234.0),
        ("1,23", 1.23),
        ("1.234,56", 1234.56),
        ("1,234.56", 1234.56),
        ("−5.2", -5.2),  # unicode minus
        ("78", 78.0),
    ],
)
def test_number_parsing(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


def test_non_numeric_returns_none():
    assert parse_number("pending") is None


def test_micro_sign_is_folded():
    assert normalize_unit_text("µg/dL") == "ug/dL"
    assert normalize_unit_text("μIU/mL") == "uIU/mL"


def test_test_name_normalization():
    assert normalize_test_name("DHEA-S") == "dhea s"
    assert normalize_test_name("Anti-Müllerian Hormone (AMH)") == "anti m llerian hormone amh"


# -- Page grounding ---------------------------------------------------------


def test_every_result_is_page_grounded(document, results):
    assert results
    for result in results:
        assert result.page_number >= 1
        assert result.char_end > result.char_start
        assert document.text[result.char_start : result.char_end] == result.evidence_text
        assert result.evidence_text.strip()


def test_values_are_attributed_to_the_correct_page(results):
    assert by_code(results, "total_testosterone").page_number == 1
    assert by_code(results, "fasting_glucose").page_number == 2


def test_unmapped_tests_are_reported_not_guessed(parser):
    doc = parser.parse("[PAGE 1]\nTSH: 2.1 mIU/L\nProlactin: 14 ng/mL\n", document_id="D-UNMAPPED")
    result = LabExtractor().extract(doc, patient_id="P")
    assert result.results == []
    assert {u.source_test_name.lower() for u in result.unmapped} == {"tsh", "prolactin"}


# -- Ultrasound reports -----------------------------------------------------


@pytest.fixture(scope="module")
def ultrasound(parser):
    doc = parser.parse(ULTRASOUND_REPORT, document_id="US-TEST")
    return doc, ReportExtractor().extract(doc, patient_id="P-US")


def test_follicle_counts_per_ovary(ultrasound):
    _doc, result = ultrasound
    counts = {f.canonical_code: f.value for f in result.findings}
    assert counts["follicle_count_right"] == 19
    assert counts["follicle_count_left"] == 14


def test_ovarian_volumes_per_side(ultrasound):
    _doc, result = ultrasound
    volumes = {f.side: f.value for f in result.findings if f.canonical_code == "ovary_volume_ml"}
    assert volumes["right"] == pytest.approx(12.4)
    assert volumes["left"] == pytest.approx(9.8)


def test_impression_is_evidence_not_a_diagnosis(ultrasound):
    _doc, result = ultrasound
    impression = next(
        f for f in result.findings if f.canonical_code == "ovarian_morphology_evidence"
    )
    assert impression.value == "present"
    codes = {f.canonical_code for f in result.findings}
    assert "pcos_binary" not in codes


def test_indeterminate_beats_a_positive_keyword(parser):
    """ "cannot be excluded" is not a positive finding."""
    doc = parser.parse(
        "[PAGE 1]\nIMPRESSION: Polycystic ovarian morphology cannot be excluded.\n",
        document_id="US-IND",
    )
    result = ReportExtractor().extract(doc, patient_id="P")
    impression = next(
        f for f in result.findings if f.canonical_code == "ovarian_morphology_evidence"
    )
    assert impression.value == "indeterminate"


def test_report_events_require_evidence(ultrasound):
    _doc, result = ultrasound
    events = findings_to_events(result.findings, modality="ultrasound_report")
    assert events
    for event in events:
        assert event.modality == "ultrasound_report"
        assert event.source_page is not None
        assert event.evidence_text
        assert event.is_model_ready is False
