"""Confirmation state machine for speech-derived assertions.

Why this exists: speech extraction is the least reliable input PRISM has, and it
is also the one that speaks in the patient's own voice. Both facts point the same
way — a human must sign off before an extraction counts as an observation.

The state machine is deliberately small and one-directional:

    proposed --confirm--> confirmed
    proposed --reject---> rejected
    proposed --edit-----> proposed (value corrected, still unconfirmed)

There is no transition that produces a confirmed event without a reviewer
identity, and :meth:`ConfirmationSession.to_events` refuses to stamp
``confirmation_status='confirmed'`` on anything that did not pass through
:meth:`confirm`. That refusal, not a convention, is what keeps unconfirmed
speech out of model-ready snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ingestion.speech.extraction import ExtractedSymptomEvent, ExtractionResult
from schemas.event import HormonalHealthEvent, Modality, Provenance
from schemas.evidence import ConfirmationBatch

ConfirmationState = Literal["proposed", "confirmed", "rejected"]

AWAITING_PATIENT = "awaiting_patient_confirmation"
AWAITING_CLINICIAN = "awaiting_clinician_confirmation"


class ConfirmationItem(BaseModel):
    """One extraction plus its review state and any reviewer correction."""

    extraction: ExtractedSymptomEvent
    state: ConfirmationState = "proposed"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    corrected_value: Any = None
    was_corrected: bool = False
    reviewer_note: str = ""

    @property
    def effective_value(self) -> Any:
        return self.corrected_value if self.was_corrected else self.extraction.value


class ConfirmationSession:
    """Holds proposals for one recording and turns confirmed ones into events."""

    def __init__(
        self,
        patient_id: str,
        recording_id: str,
        *,
        source_dataset: str | None = None,
        audio_quality: float = 1.0,
    ) -> None:
        self.patient_id = patient_id
        self.recording_id = recording_id
        self.source_dataset = source_dataset
        self.audio_quality = audio_quality
        self.items: dict[str, ConfirmationItem] = {}

    # -- Building -----------------------------------------------------------

    def propose(self, extractions: list[ExtractedSymptomEvent]) -> None:
        """Add extractions in the ``proposed`` state."""
        for extraction in extractions:
            self.items[extraction.extraction_id] = ConfirmationItem(extraction=extraction)

    @classmethod
    def from_result(
        cls,
        result: ExtractionResult,
        *,
        recording_id: str | None = None,
        source_dataset: str | None = None,
        audio_quality: float = 1.0,
    ) -> ConfirmationSession:
        session = cls(
            result.patient_id,
            recording_id or result.recording_id,
            source_dataset=source_dataset,
            audio_quality=audio_quality,
        )
        session.propose(result.events)
        return session

    # -- Transitions --------------------------------------------------------

    def confirm(self, extraction_id: str, reviewed_by: str, note: str = "") -> ConfirmationItem:
        """Mark one proposal as confirmed by a named reviewer.

        ``reviewed_by`` is required and non-empty: an anonymous confirmation is
        indistinguishable from no confirmation.
        """
        if not reviewed_by.strip():
            raise ValueError("confirm() requires a non-empty reviewer identity.")
        item = self._require(extraction_id)
        if item.state == "rejected":
            raise ValueError(f"{extraction_id}: a rejected item cannot be confirmed.")
        item.state = "confirmed"
        item.reviewed_by = reviewed_by
        item.reviewed_at = datetime.now(UTC)
        item.reviewer_note = note
        return item

    def reject(self, extraction_id: str, reviewed_by: str, note: str = "") -> ConfirmationItem:
        """Mark one proposal as rejected. Rejected items are kept, never deleted."""
        item = self._require(extraction_id)
        item.state = "rejected"
        item.reviewed_by = reviewed_by
        item.reviewed_at = datetime.now(UTC)
        item.reviewer_note = note
        return item

    def correct(self, extraction_id: str, value: Any, note: str = "") -> ConfirmationItem:
        """Record a reviewer's value correction without confirming it.

        A correction is evidence that the extractor was wrong, so it is counted
        by the user-correction-rate metric and still requires a separate
        :meth:`confirm` call.
        """
        item = self._require(extraction_id)
        item.corrected_value = value
        item.was_corrected = True
        item.reviewer_note = note
        item.state = "proposed"
        return item

    def confirm_all(self, reviewed_by: str) -> None:
        """Confirm every still-proposed item. Test and demo convenience only."""
        for extraction_id, item in self.items.items():
            if item.state == "proposed":
                self.confirm(extraction_id, reviewed_by)

    def _require(self, extraction_id: str) -> ConfirmationItem:
        item = self.items.get(extraction_id)
        if item is None:
            raise KeyError(f"unknown extraction_id '{extraction_id}'.")
        return item

    # -- Metrics ------------------------------------------------------------

    @property
    def correction_rate(self) -> float:
        """Fraction of reviewed items whose value a human had to change."""
        reviewed = [i for i in self.items.values() if i.state != "proposed"]
        if not reviewed:
            return 0.0
        return sum(1 for i in reviewed if i.was_corrected) / len(reviewed)

    # -- Conversion ---------------------------------------------------------

    def to_events(self) -> list[HormonalHealthEvent]:
        """Convert **only confirmed** items into canonical events."""
        return [
            self._to_event(item, confirmed=True)
            for item in self.items.values()
            if item.state == "confirmed"
        ]

    def pending_events(self) -> list[HormonalHealthEvent]:
        """Convert still-proposed items, marked as awaiting confirmation.

        These are model-ready-negative by construction: ``is_model_ready`` is
        False for every event this returns.
        """
        return [
            self._to_event(item, confirmed=False)
            for item in self.items.values()
            if item.state == "proposed"
        ]

    def rejected_events(self) -> list[HormonalHealthEvent]:
        return [
            self._to_event(item, confirmed=False, rejected=True)
            for item in self.items.values()
            if item.state == "rejected"
        ]

    def build_batch(self) -> ConfirmationBatch:
        """Assemble the review-UI payload."""
        return ConfirmationBatch(
            patient_id=self.patient_id,
            confirmed=self.to_events(),
            awaiting_confirmation=self.pending_events(),
            rejected=self.rejected_events(),
        )

    # -- Internals ----------------------------------------------------------

    def _to_event(
        self, item: ConfirmationItem, *, confirmed: bool, rejected: bool = False
    ) -> HormonalHealthEvent:
        extraction = item.extraction
        modality: Modality = (
            "clinician_voice" if extraction.speaker_role == "clinician" else "patient_voice"
        )
        provenance: Provenance = (
            "clinician_confirmed" if modality == "clinician_voice" else "patient_confirmed"
        )

        if rejected:
            status: str = "rejected"
        elif confirmed:
            # HormonalHealthEvent forbids confirmed + negated + uncertain: an
            # assertion that is simultaneously "no" and "maybe" has not actually
            # been resolved by the reviewer, so it stays pending instead.
            if extraction.negated and extraction.uncertain:
                status = AWAITING_CLINICIAN
            else:
                status = "confirmed"
        else:
            status = AWAITING_CLINICIAN if modality == "clinician_voice" else AWAITING_PATIENT

        return HormonalHealthEvent(
            patient_id=extraction.patient_id,
            variable_name=extraction.variable_name,
            canonical_variable_code=extraction.canonical_code,
            value=item.effective_value,
            unit=extraction.unit,
            raw_value=extraction.value,
            raw_unit=extraction.unit,
            modality=modality,
            provenance=provenance,
            extraction_confidence=extraction.extraction_confidence,
            confirmation_status=status,  # type: ignore[arg-type]
            reviewed_by=item.reviewed_by,
            reviewed_at=item.reviewed_at,
            missingness_status="observed",
            negated=extraction.negated,
            historical=extraction.historical,
            uncertain=extraction.uncertain,
            source_dataset=self.source_dataset,
            source_file_id=self.recording_id,
            source_time_start_seconds=extraction.evidence.start_seconds,
            source_time_end_seconds=extraction.evidence.end_seconds,
            evidence_text=extraction.evidence.text,
            parser_version=extraction.extractor_version,
        )


class ConfirmationSummary(BaseModel):
    """Counts used by the evaluation report and the processing manifest."""

    recording_id: str
    n_proposed: int = 0
    n_confirmed: int = 0
    n_rejected: int = 0
    n_corrected: int = 0
    correction_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    n_model_ready: int = 0


def summarize(session: ConfirmationSession) -> ConfirmationSummary:
    """Summarize a session, including how many events actually became usable."""
    states = [i.state for i in session.items.values()]
    return ConfirmationSummary(
        recording_id=session.recording_id,
        n_proposed=sum(1 for s in states if s == "proposed"),
        n_confirmed=sum(1 for s in states if s == "confirmed"),
        n_rejected=sum(1 for s in states if s == "rejected"),
        n_corrected=sum(1 for i in session.items.values() if i.was_corrected),
        correction_rate=session.correction_rate,
        n_model_ready=sum(1 for e in session.to_events() if e.is_model_ready),
    )
