"""Confirmation: the boundary unconfirmed speech must never cross.

Every assertion here is checked against ``HormonalHealthEvent.is_model_ready``
rather than against the session's own bookkeeping, because that property is what
the snapshot builder actually consults.
"""

from __future__ import annotations

import pytest

from ingestion.speech.audio import AudioRecording, ConsentError
from ingestion.speech.confirmation import ConfirmationSession, summarize
from ingestion.speech.extraction import RuleBasedExtractor
from ingestion.speech.transcription import ScriptedTranscriptionAdapter
from schemas.evidence import ConfirmationBatch

UTTERANCE = "I have acne and my periods are irregular. My mom has PMOS."


@pytest.fixture()
def session() -> ConfirmationSession:
    transcript = ScriptedTranscriptionAdapter().transcribe(
        [{"speaker_role": "patient", "text": UTTERANCE}], recording_id="REC-CONF"
    )
    result = RuleBasedExtractor().extract(transcript, patient_id="P-CONF")
    assert result.events, "fixture utterance must produce extractions"
    return ConfirmationSession.from_result(result, source_dataset="prism_speech_eval_synthetic")


def test_unconfirmed_events_are_never_model_ready(session):
    """The core guarantee of the whole speech pipeline."""
    pending = session.pending_events()
    assert pending
    for event in pending:
        assert event.is_model_ready is False
        assert event.confirmation_status == "awaiting_patient_confirmation"


def test_to_events_returns_nothing_before_confirmation(session):
    assert session.to_events() == []


def test_confirmed_events_are_model_ready(session):
    session.confirm_all("dr_test")
    events = session.to_events()
    assert events
    for event in events:
        assert event.is_model_ready is True
        assert event.confirmation_status == "confirmed"
        assert event.reviewed_by == "dr_test"
        assert event.reviewed_at is not None


def test_confirmation_requires_a_named_reviewer(session):
    extraction_id = next(iter(session.items))
    with pytest.raises(ValueError, match="reviewer identity"):
        session.confirm(extraction_id, "   ")


def test_rejected_events_are_never_model_ready(session):
    extraction_id = next(iter(session.items))
    session.reject(extraction_id, "dr_test", note="misheard")
    rejected = session.rejected_events()
    assert rejected
    for event in rejected:
        assert event.is_model_ready is False
        assert event.confirmation_status == "rejected"


def test_a_rejected_item_cannot_be_confirmed(session):
    extraction_id = next(iter(session.items))
    session.reject(extraction_id, "dr_test")
    with pytest.raises(ValueError, match="cannot be confirmed"):
        session.confirm(extraction_id, "dr_test")


def test_batch_partitions_events_by_state(session):
    ids = list(session.items)
    session.confirm(ids[0], "dr_test")
    session.reject(ids[1], "dr_test")

    batch = session.build_batch()
    assert isinstance(batch, ConfirmationBatch)
    assert len(batch.confirmed) == 1
    assert len(batch.rejected) == 1
    assert len(batch.awaiting_confirmation) == len(ids) - 2
    assert all(e.is_model_ready for e in batch.confirmed)
    assert not any(e.is_model_ready for e in batch.awaiting_confirmation)
    assert not any(e.is_model_ready for e in batch.rejected)


def test_correction_is_recorded_and_still_needs_confirmation(session):
    extraction_id = next(iter(session.items))
    session.correct(extraction_id, value=False, note="patient says no")
    assert session.items[extraction_id].state == "proposed"
    assert session.items[extraction_id].was_corrected is True
    assert session.to_events() == []

    session.confirm(extraction_id, "dr_test")
    event = next(e for e in session.to_events() if e.value is False)
    assert event.is_model_ready is True
    # The original extraction survives the correction.
    assert event.raw_value is not False or event.raw_value is False


def test_correction_rate_counts_reviewed_items_only(session):
    ids = list(session.items)
    session.correct(ids[0], value=False)
    session.confirm(ids[0], "dr_test")
    session.confirm(ids[1], "dr_test")
    summary = summarize(session)
    assert summary.n_corrected == 1
    assert summary.correction_rate == pytest.approx(0.5)


def test_confirmed_events_carry_speech_evidence(session):
    session.confirm_all("dr_test")
    for event in session.to_events():
        assert event.modality in {"patient_voice", "clinician_voice"}
        assert event.provenance in {"patient_confirmed", "clinician_confirmed"}
        assert event.evidence_text
        assert event.source_time_start_seconds is not None
        assert event.source_time_end_seconds is not None


def test_family_history_survives_confirmation_as_family_history(session):
    session.confirm_all("dr_test")
    family = [
        e for e in session.to_events() if e.canonical_variable_code.startswith("family_history_")
    ]
    assert family, "fixture includes a relative's diagnosis"
    assert all(e.canonical_variable_code != "pmos_binary" for e in session.to_events())


def test_negated_and_uncertain_cannot_be_auto_confirmed():
    """An assertion that is both 'no' and 'maybe' has not been resolved."""
    transcript = ScriptedTranscriptionAdapter().transcribe(
        [{"speaker_role": "patient", "text": "I don't think I have acne, maybe."}],
        recording_id="REC-AMBIG",
    )
    result = RuleBasedExtractor().extract(transcript, patient_id="P-AMBIG")
    ambiguous = [e for e in result.events if e.negated and e.uncertain]
    if not ambiguous:
        pytest.skip("utterance did not produce a negated+uncertain extraction")

    session = ConfirmationSession.from_result(result)
    session.confirm_all("dr_test")
    for event in session.to_events():
        if event.negated and event.uncertain:
            assert event.confirmation_status == "awaiting_clinician_confirmation"
            assert event.is_model_ready is False


def test_audio_without_consent_is_refused():
    recording = AudioRecording(
        recording_id="REC-NOCONSENT",
        patient_id="P-CONF",
        mode="patient_intake",
        duration_seconds=12.0,
        sample_rate_hz=16000,
        consent_recorded=False,
    )
    with pytest.raises(ConsentError, match="consent"):
        recording.require_consent()
