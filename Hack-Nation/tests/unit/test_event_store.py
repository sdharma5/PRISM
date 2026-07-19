"""Event store behaviour: append-only, conflict-preserving, review-gated."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from event_store.conflict_resolution import (
    RELATIVE_DIFFERENCE_THRESHOLD,
    detect_conflicts,
    select_winner,
)
from event_store.provenance import PROVENANCE_TRUST_ORDER, provenance_rank, trace_event
from event_store.queries import (
    by_confirmation_status,
    by_modality,
    by_patient,
    by_time_window,
    by_variable_code,
)
from event_store.serialization import (
    events_from_dataframe,
    events_from_jsonl,
    events_to_dataframe,
    events_to_jsonl,
)
from event_store.store import AppendOnlyViolationError, EventStore
from schemas.event import HormonalHealthEvent

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
PATIENT = "pcos_tabular_public:FAKE-001"


def event(**overrides) -> HormonalHealthEvent:
    payload = {
        "patient_id": PATIENT,
        "variable_name": "Total testosterone",
        "canonical_variable_code": "total_testosterone",
        "value": 45.0,
        "unit": "ng/dL",
        "raw_value": 45.0,
        "raw_unit": "ng/dL",
        "modality": "laboratory",
        "provenance": "dataset_provided",
        "extraction_confidence": 1.0,
        "confirmation_status": "not_required",
        "observed_at": NOW,
        "source_dataset": "pcos_tabular_public",
    }
    payload.update(overrides)
    return HormonalHealthEvent(**payload)


# -- Append-only -----------------------------------------------------------


def test_append_stores_and_returns_the_event():
    store = EventStore()
    stored = store.append(event())
    assert len(store) == 1
    assert store.get(PATIENT) == [stored]


def test_extend_returns_the_count():
    store = EventStore()
    assert store.extend([event(), event()]) == 2
    assert len(store) == 2


def test_appending_the_same_event_id_twice_is_refused():
    store = EventStore()
    original = store.append(event())
    with pytest.raises(AppendOnlyViolationError):
        store.append(original)


def test_store_has_no_delete_or_update_api():
    # The ledger is the audit trail; a mutation API would make it unfalsifiable.
    for forbidden in ("delete", "remove", "update", "pop", "clear"):
        assert not hasattr(EventStore, forbidden)


def test_events_view_is_immutable():
    store = EventStore()
    store.append(event())
    assert isinstance(store.events, tuple)


def test_confirmation_change_appends_a_revision_and_keeps_the_original():
    store = EventStore()
    original = store.append(event(confirmation_status="awaiting_clinician_confirmation"))
    revision = store.append_confirmation_revision(
        str(original.event_id), confirmation_status="confirmed", reviewed_by="dr-fake"
    )

    assert len(store) == 2
    assert revision.event_id != original.event_id
    # The original is untouched: its status still reads as it did before review.
    assert store.get_by_id(str(original.event_id)).confirmation_status == (
        "awaiting_clinician_confirmation"
    )
    assert revision.confirmation_status == "confirmed"
    assert revision.reviewed_by == "dr-fake"


def test_snapshot_uses_the_revision_not_the_superseded_original():
    store = EventStore()
    original = store.append(event(confirmation_status="awaiting_clinician_confirmation"))
    revision = store.append_confirmation_revision(
        str(original.event_id), confirmation_status="confirmed", reviewed_by="dr-fake"
    )
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))

    assert snapshot.values["total_testosterone"].source_event_id == str(revision.event_id)
    assert str(original.event_id) in snapshot.excluded_event_ids


# -- Snapshot gating -------------------------------------------------------


@pytest.mark.parametrize(
    "status", ["awaiting_patient_confirmation", "awaiting_clinician_confirmation", "rejected"]
)
def test_snapshot_excludes_unconfirmed_events(status):
    store = EventStore()
    unconfirmed = store.append(event(confirmation_status=status))
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))

    assert snapshot.values == {}
    assert str(unconfirmed.event_id) in snapshot.excluded_event_ids
    assert status in snapshot.exclusion_reasons[str(unconfirmed.event_id)]


def test_snapshot_excludes_model_inferred_events_awaiting_review():
    store = EventStore()
    store.append(
        event(
            provenance="model_inferred",
            confirmation_status="awaiting_clinician_confirmation",
            extraction_confidence=0.99,
        )
    )
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))
    assert snapshot.values == {}


def test_snapshot_includes_confirmed_and_not_required_events():
    store = EventStore()
    store.append(event())
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))
    assert snapshot.values["total_testosterone"].value == 45.0
    assert snapshot.missingness_mask["total_testosterone"] is False


def test_snapshot_excludes_missing_events_rather_than_zero_filling():
    store = EventStore()
    store.append(
        event(
            canonical_variable_code="fasting_glucose",
            value=None,
            unit=None,
            missingness_status="not_collected",
        )
    )
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))
    assert "fasting_glucose" not in snapshot.values
    assert 0 not in [v.value for v in snapshot.values.values()]


def test_snapshot_excludes_events_observed_after_as_of():
    store = EventStore()
    future = store.append(event(observed_at=NOW + timedelta(days=30)))
    snapshot = store.build_snapshot(PATIENT, as_of=NOW)
    assert snapshot.exclusion_reasons[str(future.event_id)] == "observed_at is after as_of"


def test_snapshot_respects_a_modality_allowlist():
    store = EventStore()
    store.append(event())
    store.append(
        event(canonical_variable_code="acne", value=True, unit=None, modality="questionnaire")
    )
    snapshot = store.build_snapshot(
        PATIENT, as_of=NOW + timedelta(days=1), include_modalities=["laboratory"]
    )
    assert set(snapshot.values) == {"total_testosterone"}
    assert snapshot.included_modalities == ["laboratory"]


def test_snapshot_records_recency_in_days():
    store = EventStore()
    store.append(event(observed_at=NOW - timedelta(days=10)))
    snapshot = store.build_snapshot(PATIENT, as_of=NOW)
    assert snapshot.values["total_testosterone"].recency_days == pytest.approx(10.0)


def test_snapshot_never_spans_patients():
    store = EventStore()
    store.append(event())
    store.append(event(patient_id="nhanes_2021_2023:12345", value=90.0))
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))
    assert snapshot.values["total_testosterone"].value == 45.0
    assert snapshot.patient_id == PATIENT


def test_snapshot_coverage_and_observed_codes():
    store = EventStore()
    store.append(event())
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))
    assert snapshot.observed_codes() == ["total_testosterone"]
    assert snapshot.coverage(["total_testosterone", "fasting_glucose"]) == 0.5


# -- Conflicts -------------------------------------------------------------


def test_disagreeing_values_are_both_kept_and_a_conflict_is_raised():
    store = EventStore()
    low = store.append(event(value=45.0, observed_at=NOW - timedelta(days=2)))
    high = store.append(
        event(value=90.0, provenance="clinician_confirmed", confirmation_status="confirmed")
    )

    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))

    assert len(store) == 2, "neither event may be removed from the ledger"
    assert store.get_by_id(str(low.event_id)) is not None
    assert store.get_by_id(str(high.event_id)) is not None

    assert len(snapshot.conflicts) == 1
    conflict = snapshot.conflicts[0]
    assert conflict.conflict_type == "value_disagreement"
    assert set(conflict.event_ids) == {str(low.event_id), str(high.event_id)}
    assert conflict.requires_human_review is True

    selected = snapshot.values["total_testosterone"]
    assert selected.n_candidates == 2
    # Higher provenance trust wins even though the other event is also valid.
    assert selected.source_event_id == str(high.event_id)
    assert str(low.event_id) in snapshot.excluded_event_ids
    assert "not_selected" in snapshot.exclusion_reasons[str(low.event_id)]
    assert snapshot.warnings


def test_conflict_is_not_overwritten_by_a_later_append():
    store = EventStore()
    store.append(event(value=45.0))
    store.append(event(value=90.0))
    store.append(event(value=200.0))
    snapshot = store.build_snapshot(PATIENT, as_of=NOW + timedelta(days=1))
    assert len(store) == 3
    assert len(snapshot.conflicts) == 3  # all pairs surfaced, none suppressed


def test_values_within_the_threshold_do_not_conflict():
    a = event(value=100.0)
    b = event(value=100.0 * (1 + RELATIVE_DIFFERENCE_THRESHOLD * 0.5))
    assert detect_conflicts([a, b]) == []


def test_values_beyond_the_threshold_conflict():
    a = event(value=100.0)
    b = event(value=100.0 * (1 + RELATIVE_DIFFERENCE_THRESHOLD * 2))
    assert detect_conflicts([a, b])[0].conflict_type == "value_disagreement"


def test_unit_disagreement_is_detected():
    a = event(value=45.0, unit="ng/dL")
    b = event(value=45.0, unit="nmol/L")
    assert detect_conflicts([a, b])[0].conflict_type == "unit_disagreement"


def test_presence_versus_negation_is_detected():
    a = event(canonical_variable_code="acne", value=True, unit=None, modality="questionnaire")
    b = event(
        canonical_variable_code="acne",
        value=True,
        unit=None,
        modality="questionnaire",
        negated=True,
    )
    assert detect_conflicts([a, b])[0].conflict_type == "presence_vs_negation"


def test_duplicate_measurement_is_detected_and_needs_no_review():
    conflict = detect_conflicts([event(), event()])[0]
    assert conflict.conflict_type == "duplicate_measurement"
    assert conflict.requires_human_review is False


def test_temporal_disagreement_is_detected():
    a = event(value=45.0, observed_at=NOW)
    b = event(value=45.0, observed_at=NOW - timedelta(days=800))
    assert detect_conflicts([a, b])[0].conflict_type == "temporal_disagreement"


def test_unobserved_events_cannot_conflict():
    a = event(value=None, unit=None, missingness_status="not_collected")
    assert detect_conflicts([a, event()]) == []


def test_select_winner_prefers_trust_then_recency():
    old_trusted = event(
        provenance="clinician_confirmed",
        confirmation_status="confirmed",
        observed_at=NOW - timedelta(days=100),
    )
    new_untrusted = event(provenance="dataset_provided", observed_at=NOW)
    assert select_winner([new_untrusted, old_trusted]) is old_trusted

    newer = event(observed_at=NOW)
    older = event(observed_at=NOW - timedelta(days=5))
    assert select_winner([older, newer]) is newer


def test_provenance_trust_order_is_total_and_ranks_models_last():
    assert provenance_rank("clinician_confirmed") < provenance_rank("dataset_provided")
    assert provenance_rank("dataset_provided") < provenance_rank("model_inferred")
    assert PROVENANCE_TRUST_ORDER[-1] == "model_inferred"
    assert provenance_rank("nonsense") == len(PROVENANCE_TRUST_ORDER)


# -- Queries ---------------------------------------------------------------


def test_queries_filter_without_mutating_the_store():
    store = EventStore()
    store.append(event())
    store.append(
        event(
            canonical_variable_code="acne",
            value=True,
            unit=None,
            modality="questionnaire",
            confirmation_status="awaiting_patient_confirmation",
            observed_at=NOW - timedelta(days=400),
        )
    )
    events = store.events

    assert len(by_patient(events, PATIENT)) == 2
    assert len(by_patient(events, "nobody")) == 0
    assert len(by_modality(events, "laboratory")) == 1
    assert len(by_modality(events, ["laboratory", "questionnaire"])) == 2
    assert len(by_variable_code(events, "acne")) == 1
    assert len(by_confirmation_status(events, "not_required")) == 1
    assert len(by_time_window(events, start=NOW - timedelta(days=1))) == 1
    assert len(store) == 2


def test_by_time_window_excludes_undated_events_by_default():
    events = [event(observed_at=None)]
    assert by_time_window(events, start=NOW - timedelta(days=1)) == []
    assert len(by_time_window(events, start=NOW - timedelta(days=1), include_undated=True)) == 1


# -- Provenance ------------------------------------------------------------


def test_trace_event_returns_the_source_receipt():
    store = EventStore()
    stored = store.append(event(source_file_id="pcos_tabular_tiny.csv", source_file_hash="abc123"))
    trace = trace_event(str(stored.event_id), store.events)
    assert trace["source_file_id"] == "pcos_tabular_tiny.csv"
    assert trace["source_file_hash"] == "abc123"
    assert trace["raw_value"] == 45.0
    assert trace["raw_unit"] == "ng/dL"


def test_trace_event_raises_for_an_unknown_id():
    with pytest.raises(KeyError):
        trace_event("00000000-0000-0000-0000-000000000000", [])


# -- Serialization ---------------------------------------------------------


def test_jsonl_round_trip(tmp_path):
    events = [event(), event(value=90.0)]
    path = events_to_jsonl(events, tmp_path / "events.jsonl")
    restored = events_from_jsonl(path)
    assert [str(e.event_id) for e in restored] == [str(e.event_id) for e in events]
    assert restored[0].raw_unit == "ng/dL"


def test_dataframe_round_trip():
    events = [
        event(),
        event(canonical_variable_code="acne", value=True, unit=None, modality="questionnaire"),
    ]
    frame = events_to_dataframe(events)
    assert len(frame) == 2
    assert "canonical_variable_code" in frame.columns
    restored = events_from_dataframe(frame)
    assert {e.canonical_variable_code for e in restored} == {"total_testosterone", "acne"}


def test_empty_dataframe_still_has_the_schema_columns():
    frame = events_to_dataframe([])
    assert "patient_id" in frame.columns
    assert frame.empty


def test_store_persists_appends_to_its_path(tmp_path):
    path = tmp_path / "ledger.jsonl"
    store = EventStore(path)
    store.append(event())
    store.append(event(value=90.0))
    reloaded = EventStore.load(path)
    assert len(reloaded) == 2
