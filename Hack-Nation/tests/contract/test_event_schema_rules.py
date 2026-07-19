"""Contract tests for every validation rule in schemas/event.py.

Each rule is tested in both directions: the violating case must raise, and the
compliant case must construct. A one-directional test would pass against a
validator that rejects everything.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from schemas.event import (
    EVIDENCE_REQUIRED_MODALITIES,
    SCHEMA_VERSION,
    UNREVIEWED_PROVENANCE,
    HormonalHealthEvent,
)

pytestmark = pytest.mark.contract


def make_event(**overrides) -> HormonalHealthEvent:
    payload = {
        "patient_id": "fake:1",
        "variable_name": "Total testosterone",
        "canonical_variable_code": "total_testosterone",
        "value": 45.0,
        "unit": "ng/dL",
        "raw_value": "45.0",
        "raw_unit": "ng/dL",
        "modality": "laboratory",
        "provenance": "dataset_provided",
        "extraction_confidence": 1.0,
        "confirmation_status": "not_required",
    }
    payload.update(overrides)
    return HormonalHealthEvent(**payload)


# -- Rule: observed requires a value --------------------------------------


def test_observed_without_value_is_rejected():
    with pytest.raises(ValidationError, match="requires a value"):
        make_event(value=None, missingness_status="observed")


def test_observed_with_value_is_accepted():
    assert make_event(missingness_status="observed").value == 45.0


# -- Rule: non-observed must not carry a value ----------------------------


@pytest.mark.parametrize(
    "status",
    [
        "not_collected",
        "not_available",
        "not_applicable",
        "extraction_failed",
        "intentionally_masked",
    ],
)
def test_non_observed_with_a_value_is_rejected(status):
    with pytest.raises(ValidationError, match="must not carry a value"):
        make_event(value=45.0, missingness_status=status)


@pytest.mark.parametrize(
    "status",
    [
        "not_collected",
        "not_available",
        "not_applicable",
        "extraction_failed",
        "intentionally_masked",
    ],
)
def test_non_observed_without_a_value_is_accepted(status):
    event = make_event(value=None, unit=None, missingness_status=status)
    assert event.value is None
    # Raw provenance survives even when the canonical value does not.
    assert event.raw_value == "45.0"


def test_missing_value_is_never_coerced_to_zero():
    event = make_event(value=None, unit=None, missingness_status="not_collected")
    assert event.value is None
    assert event.value != 0


# -- Rule: laboratory events require a unit -------------------------------


def test_laboratory_event_without_any_unit_is_rejected():
    with pytest.raises(ValidationError, match="require a unit"):
        make_event(unit=None, raw_unit=None)


def test_laboratory_event_with_only_a_raw_unit_is_accepted():
    assert make_event(unit=None, raw_unit="dimensionless").raw_unit == "dimensionless"


def test_non_laboratory_event_needs_no_unit():
    event = make_event(
        modality="questionnaire",
        canonical_variable_code="acne",
        value=True,
        unit=None,
        raw_unit=None,
    )
    assert event.unit is None


def test_unobserved_laboratory_event_needs_no_unit():
    event = make_event(value=None, unit=None, raw_unit=None, missingness_status="not_collected")
    assert event.missingness_status == "not_collected"


# -- Rule: unreviewed provenance cannot self-confirm -----------------------


@pytest.mark.parametrize("provenance", sorted(UNREVIEWED_PROVENANCE))
def test_unreviewed_provenance_cannot_be_confirmed_without_a_reviewer(provenance):
    with pytest.raises(ValidationError, match="human review is required"):
        make_event(provenance=provenance, confirmation_status="confirmed", reviewed_by=None)


@pytest.mark.parametrize("provenance", sorted(UNREVIEWED_PROVENANCE))
def test_unreviewed_provenance_may_be_confirmed_with_a_reviewer(provenance):
    event = make_event(
        provenance=provenance,
        confirmation_status="confirmed",
        reviewed_by="dr-fake",
        reviewed_at=datetime.now(UTC),
    )
    assert event.reviewed_by == "dr-fake"


@pytest.mark.parametrize("provenance", sorted(UNREVIEWED_PROVENANCE))
def test_unreviewed_provenance_awaiting_review_is_fine(provenance):
    event = make_event(provenance=provenance, confirmation_status="awaiting_clinician_confirmation")
    assert event.is_model_ready is False


# -- Rule: evidence-bearing modalities need a location ---------------------


@pytest.mark.parametrize("modality", sorted(EVIDENCE_REQUIRED_MODALITIES))
def test_evidence_modality_without_location_is_rejected(modality):
    with pytest.raises(ValidationError, match="requires"):
        make_event(modality=modality, unit=None, raw_unit=None)


@pytest.mark.parametrize("modality", sorted(EVIDENCE_REQUIRED_MODALITIES))
@pytest.mark.parametrize(
    "location",
    [
        {"evidence_text": "patient reports irregular cycles"},
        {"source_page": 3},
        {"source_time_start_seconds": 12.5},
    ],
)
def test_evidence_modality_with_any_location_is_accepted(modality, location):
    event = make_event(modality=modality, **location)
    assert event.modality == modality


def test_evidence_modality_unobserved_needs_no_location():
    event = make_event(
        modality="clinical_document",
        value=None,
        unit=None,
        missingness_status="extraction_failed",
    )
    assert event.missingness_status == "extraction_failed"


# -- Rule: confirmed + negated + uncertain is contradictory ----------------


def test_confirmed_negated_and_uncertain_is_rejected():
    with pytest.raises(ValidationError, match="simultaneously confirmed, negated and uncertain"):
        make_event(
            provenance="clinician_confirmed",
            confirmation_status="confirmed",
            negated=True,
            uncertain=True,
        )


def test_confirmed_and_negated_without_uncertainty_is_accepted():
    event = make_event(
        provenance="clinician_confirmed", confirmation_status="confirmed", negated=True
    )
    assert event.asserts_presence is False


def test_negated_and_uncertain_while_awaiting_review_is_accepted():
    event = make_event(
        confirmation_status="awaiting_clinician_confirmation", negated=True, uncertain=True
    )
    assert event.uncertain is True


# -- Field constraints -----------------------------------------------------


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
def test_extraction_confidence_outside_zero_one_is_rejected(confidence):
    with pytest.raises(ValidationError):
        make_event(extraction_confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_extraction_confidence_within_zero_one_is_accepted(confidence):
    assert make_event(extraction_confidence=confidence).extraction_confidence == confidence


def test_unknown_modality_is_rejected():
    with pytest.raises(ValidationError):
        make_event(modality="tea_leaves")


def test_unknown_provenance_is_rejected():
    with pytest.raises(ValidationError):
        make_event(provenance="vibes")


def test_schema_version_is_stamped():
    assert make_event().schema_version == SCHEMA_VERSION


# -- Semantics helpers -----------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("confirmed", True),
        ("not_required", True),
        ("awaiting_patient_confirmation", False),
        ("awaiting_clinician_confirmation", False),
        ("rejected", False),
    ],
)
def test_is_model_ready_tracks_confirmation_status(status, expected):
    kwargs = {"provenance": "clinician_confirmed"} if status == "confirmed" else {}
    assert make_event(confirmation_status=status, **kwargs).is_model_ready is expected


def test_is_model_ready_is_false_for_unobserved_events():
    event = make_event(value=None, unit=None, missingness_status="not_collected")
    assert event.is_model_ready is False


def test_asserts_presence_is_false_for_historical_findings():
    assert make_event(historical=True).asserts_presence is False
    assert make_event().asserts_presence is True


def test_raw_value_and_unit_are_preserved_alongside_canonical():
    event = make_event(value=28.818, unit="ng/dL", raw_value=1.0, raw_unit="nmol/L")
    assert (event.raw_value, event.raw_unit) == (1.0, "nmol/L")
    assert (event.value, event.unit) == (28.818, "ng/dL")
