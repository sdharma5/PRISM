"""PMOS tabular adapter tests against a tiny, obviously fake fixture."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ingestion.base import file_checksum
from ingestion.tabular_pmos.cleaning import clean_value, coerce_bool, coerce_numeric
from ingestion.tabular_pmos.loader import PmosTabularAdapter
from ingestion.tabular_pmos.mapping import (
    EXCLUDED_COLUMNS,
    SOURCE_COLUMN_MAP,
    canonical_code_for,
    modality_for,
    normalize_column,
)
from ingestion.tabular_pmos.validation import unaccounted_columns, validate_columns
from registry.loader import load_variable_registry

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pmos_tabular_tiny.csv"


@pytest.fixture
def adapter() -> PmosTabularAdapter:
    return PmosTabularAdapter(dataset_version="tiny-fixture-v0")


@pytest.fixture
def events(adapter: PmosTabularAdapter):
    return adapter.run(FIXTURE)


# -- Mapping ---------------------------------------------------------------


def test_column_lookup_is_case_and_whitespace_insensitive():
    assert canonical_code_for(" Age (yrs)") == "age"
    assert canonical_code_for("age (yrs)") == "age"
    assert canonical_code_for("AGE   (YRS)  ") == "age"
    # "Cycle length(days)" is menses_duration, not cycle_length: the source
    # column's values centre near 5 days. See SOURCE_COLUMN_MAP for why.
    assert canonical_code_for("Cycle length(days)") == "menses_duration"
    assert canonical_code_for("cycle length(days)") == "menses_duration"
    assert canonical_code_for("FSH(mIU/mL)") == "follicle_stimulating_hormone"
    assert canonical_code_for("hair growth(Y/N)") == "hair_growth_face"
    assert canonical_code_for("PMOS (Y/N)") == "pmos_binary"


def test_unmapped_column_returns_none():
    assert canonical_code_for("Favourite Colour") is None


def test_normalize_column_collapses_whitespace():
    assert normalize_column("  BP _Systolic  (mmHg) ") == "bp _systolic (mmHg)".lower()


def test_every_mapped_code_exists_in_the_variable_registry():
    registry = load_variable_registry().variables
    unknown = sorted({c for c in SOURCE_COLUMN_MAP.values() if c not in registry})
    assert unknown == []


def test_every_exclusion_has_a_documented_reason():
    assert all(reason.strip() for reason in EXCLUDED_COLUMNS.values())
    assert all(len(reason) > 15 for reason in EXCLUDED_COLUMNS.values())


def test_modality_is_inferred_per_variable():
    assert modality_for("follicle_stimulating_hormone") == "laboratory"
    assert modality_for("cycle_length") == "menstrual_history"
    assert modality_for("menses_duration") == "menstrual_history"
    assert modality_for("follicle_count_left") == "ultrasound_report"
    assert modality_for("hair_growth_face") == "questionnaire"
    # Anything unknown defaults to the conservative, non-clinical modality.
    assert modality_for("brand_new_variable") == "questionnaire"


# -- Validation ------------------------------------------------------------


def test_fixture_has_no_unaccounted_columns():
    with FIXTURE.open() as fh:
        columns = next(csv.reader(fh))
    unaccounted = [c for c in unaccounted_columns(columns) if c != "Patient File No."]
    assert unaccounted == []


def test_unknown_column_is_a_validation_error():
    errors = validate_columns(["Patient File No.", "Mystery Column"], "Patient File No.")
    assert any("Mystery Column" in e for e in errors)


def test_missing_id_column_is_a_validation_error():
    errors = validate_columns(["BMI"], "Patient File No.")
    assert any("Missing patient id column" in e for e in errors)


def test_fixture_validates_cleanly(adapter: PmosTabularAdapter):
    raw = adapter.load_raw(FIXTURE)
    assert adapter.validate_raw(raw) == []


# -- Cleaning --------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Y", True), ("y", True), ("1", True), ("N", False), ("n", False), ("0", False)],
)
def test_yes_no_cells_become_bools(raw, expected):
    assert coerce_bool(raw) is expected


def test_unrecognised_yes_no_cell_is_not_guessed():
    assert coerce_bool("maybe") is None


@pytest.mark.parametrize(("raw", "expected"), [("12.5", 12.5), (" 3 ", 3.0), ("1,200", 1200.0)])
def test_numeric_cells_are_coerced(raw, expected):
    assert coerce_numeric(raw) == expected


@pytest.mark.parametrize("raw", ["", "NA", "n/a", "-", "?"])
def test_null_tokens_become_not_collected(raw):
    result = clean_value("bmi", raw)
    assert result.value is None
    assert result.missingness_status == "not_collected"


def test_out_of_range_value_is_recorded_not_repaired():
    result = clean_value("age", "999")
    assert result.value is None, "an implausible age must not be clipped into plausibility"
    assert result.missingness_status == "not_available"
    assert "valid_range" in result.reason


def test_cleaning_skips_range_check_when_units_differ():
    # 28 inches is a healthy waist; comparing it to the centimetre range would
    # wrongly reject it, so the check is deferred to post-conversion.
    assert clean_value("waist_circumference", "28", source_unit="in").ok


def test_binary_variable_is_typed_as_bool():
    assert clean_value("acne", "Y").value is True


def test_cycle_regularity_maps_onto_registry_categories():
    assert clean_value("cycle_regularity", "R").value == "regular"
    assert clean_value("cycle_regularity", "I").value == "irregular"
    assert clean_value("cycle_regularity", "?").missingness_status == "not_collected"


# -- Transform -------------------------------------------------------------


def test_twelve_fake_patients_are_ingested(adapter: PmosTabularAdapter, events):
    assert adapter.n_source_records == 12
    assert len({e.patient_id for e in events}) == 12


def test_patient_ids_are_namespaced_by_dataset(events):
    assert all(e.patient_id.startswith("pmos_tabular_public:") for e in events)


def test_every_event_is_dataset_provided_and_not_required(events):
    assert {e.provenance for e in events} == {"dataset_provided"}
    assert {e.confirmation_status for e in events} == {"not_required"}


def test_modalities_span_the_expected_set(events):
    assert {e.modality for e in events} == {
        "laboratory",
        "questionnaire",
        "menstrual_history",
        "ultrasound_report",
    }


def test_raw_value_and_unit_are_preserved(events):
    testosterone = [
        e
        for e in events
        if e.canonical_variable_code == "total_testosterone" and e.patient_id.endswith("FAKE-001")
    ][0]
    assert testosterone.value == 41.0
    assert testosterone.unit == "ng/dL"
    assert testosterone.raw_value == "41.0"
    assert testosterone.raw_unit == "ng/dL"


def test_inch_measurements_are_converted_to_centimetres(events):
    waist = [
        e
        for e in events
        if e.canonical_variable_code == "waist_circumference" and e.patient_id.endswith("FAKE-001")
    ][0]
    assert waist.value == pytest.approx(32 * 2.54)
    assert waist.unit == "cm"
    assert waist.raw_value == "32"
    assert waist.raw_unit == "in"


def test_yes_no_columns_are_emitted_as_bools(events):
    hirsutism = [
        e
        for e in events
        if e.canonical_variable_code == "hair_growth_face" and e.patient_id.endswith("FAKE-002")
    ][0]
    assert hirsutism.value is True


def test_empty_cells_become_not_collected_never_zero(events):
    weight = [
        e
        for e in events
        if e.canonical_variable_code == "weight" and e.patient_id.endswith("FAKE-007")
    ][0]
    assert weight.value is None
    assert weight.missingness_status == "not_collected"
    assert weight.value != 0


def test_out_of_range_value_becomes_not_available_and_is_logged(adapter, events):
    age = [
        e
        for e in events
        if e.canonical_variable_code == "age" and e.patient_id.endswith("FAKE-010")
    ][0]
    assert age.missingness_status == "not_available"
    assert age.value is None
    assert age.raw_value == "999", "the implausible source number is still preserved"
    assert any("age" in d.detail for d in adapter.dropped_records)


def test_unparseable_yes_no_value_becomes_not_available_and_is_logged(adapter, events):
    hair = [
        e
        for e in events
        if e.canonical_variable_code == "hair_growth_face" and e.patient_id.endswith("FAKE-011")
    ][0]
    assert hair.missingness_status == "not_available"
    assert any("hair_growth_face" in d.detail for d in adapter.dropped_records)


def test_no_event_carries_a_value_when_not_observed(events):
    assert all(e.value is None for e in events if e.missingness_status != "observed")


def test_excluded_columns_produce_no_events(events):
    with FIXTURE.open() as fh:
        columns = next(csv.reader(fh))
    expected = {code for c in columns if (code := canonical_code_for(c)) is not None}
    assert {e.canonical_variable_code for e in events} == expected
    assert "blood_group" not in expected


def test_registry_refuses_a_prohibited_use():
    with pytest.raises(PermissionError):
        PmosTabularAdapter(use="prospective_clinical_deployment")


# -- Manifest --------------------------------------------------------------


def test_manifest_records_checksums_and_counts(adapter, events):
    manifest = adapter.build_manifest()
    assert manifest.dataset_id == "pmos_tabular_public"
    assert manifest.dataset_version == "tiny-fixture-v0"
    assert manifest.file_checksums["pmos_tabular_tiny.csv"] == file_checksum(FIXTURE)
    assert manifest.n_source_records == 12
    assert manifest.n_events_emitted == len(events)
    assert manifest.n_dropped >= 2
    assert manifest.excluded_source_columns


def test_unit_conversions_are_counted(adapter, events):
    manifest = adapter.build_manifest()
    key = "waist_circumference:in->cm"
    assert manifest.unit_conversions_applied[key] >= 10


def test_write_manifest_artifacts_writes_every_expected_file(adapter, events, tmp_path):
    written = adapter.write_manifest_artifacts(tmp_path, events=events)
    expected = {
        "raw_manifest.json",
        "file_checksums.json",
        "variable_mapping.json",
        "validation_report.json",
        "dropped_records.csv",
        "processing_config.yaml",
        "processed_manifest.json",
    }
    assert set(written) == expected
    assert all(path.exists() for path in written.values())

    checksums = json.loads(written["file_checksums.json"].read_text())
    assert checksums["pmos_tabular_tiny.csv"] == file_checksum(FIXTURE)

    with written["dropped_records.csv"].open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["record_index", "reason", "detail"]
    assert len(rows) >= 3

    mapping = json.loads(written["variable_mapping.json"].read_text())
    assert mapping["excluded_source_columns"]["Blood Group"]


def test_file_checksum_is_stable_and_content_sensitive(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello")
    first = file_checksum(a)
    assert first == file_checksum(a)
    a.write_text("hello!")
    assert file_checksum(a) != first
