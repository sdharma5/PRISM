"""What a new patient submits, before any encoder has run.

The bundle is deliberately permissive about *what* arrives and strict about
*whose* it is. Every member carries the same ``patient_id``, checked on
construction: a bundle that mixes two people would produce a report that reads
as one person's profile, which is the single most dangerous failure this system
can have.

Speech and documents are **ingestion**, not prediction. They appear here as raw
inputs and leave as :class:`HormonalHealthEvent` objects that feed the static
clinical encoder alongside questionnaire answers. They never reach the adapter
as their own diagnostic branch -- see prompt_4 section 8 and ADR-002.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.event import HormonalHealthEvent
from schemas.imaging import UltrasoundStudyMetadata
from schemas.temporal import ParticipantDay

__all__ = [
    "DocumentInput",
    "PatientDataBundle",
    "SpeechInput",
    "TemporalInput",
    "UltrasoundInput",
]


class UltrasoundInput(BaseModel):
    """One ultrasound acquisition: a frame, a cine loop, or a volume."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: UltrasoundStudyMetadata
    #: ``(H, W)`` frame, ``(T, H, W)`` cine loop, or ``(D, H, W)`` volume. The
    #: acquisition mode is inferred from rank plus ``metadata.is_3d``, because a
    #: 3-rank array is ambiguous between a loop and a volume on shape alone.
    pixels: Any = None

    @property
    def acquisition_mode(self) -> str:
        array = np.asarray(self.pixels)
        if array.ndim == 2:
            return "single_frame"
        if array.ndim == 3:
            return "volume_3d" if self.metadata.is_3d else "cine_loop"
        return "unknown"


class TemporalInput(BaseModel):
    """A longitudinal series: wearables, CGM, or repeated hormone measurements."""

    participant_days: list[ParticipantDay] = Field(default_factory=list)

    @property
    def n_days(self) -> int:
        return len(self.participant_days)


class SpeechInput(BaseModel):
    """A spoken history. Converted to events before any model sees it."""

    #: Transcript text. Audio paths are accepted by the ingestion layer, not here:
    #: this package never performs ASR, it consumes what ingestion produced.
    transcript: str = ""
    recorded_at: str | None = None
    source_id: str | None = None


class DocumentInput(BaseModel):
    """A clinical document (lab report, ultrasound report) as extracted text."""

    text: str = ""
    document_type: str = "unknown"
    reported_at: str | None = None
    source_id: str | None = None


class PatientDataBundle(BaseModel):
    """Everything one new patient has provided."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    patient_id: str

    clinical_events: list[HormonalHealthEvent] = Field(default_factory=list)
    ultrasound_inputs: list[UltrasoundInput] = Field(default_factory=list)
    temporal_series: TemporalInput | None = None

    speech_recordings: list[SpeechInput] = Field(default_factory=list)
    documents: list[DocumentInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_patient_only(self) -> PatientDataBundle:
        """Refuse a bundle whose members disagree about whose data this is."""
        foreign = (
            {
                event.patient_id
                for event in self.clinical_events
                if getattr(event, "patient_id", self.patient_id) != self.patient_id
            }
            | {
                study.metadata.patient_id
                for study in self.ultrasound_inputs
                if study.metadata.patient_id != self.patient_id
            }
            # Temporal days identify their subject as `participant_id`, so they
            # need their own comparison -- there is no `patient_id` to match on.
            | {
                day.participant_id
                for day in (self.temporal_series.participant_days if self.temporal_series else [])
                if day.participant_id != self.patient_id
            }
        )
        if foreign:
            raise ValueError(
                f"Bundle for patient '{self.patient_id}' contains data belonging to "
                f"{sorted(foreign)}. Combining them would produce one report describing "
                "more than one person."
            )
        return self

    def has_static_input(self) -> bool:
        """True when anything can populate the static clinical encoder.

        Speech and documents count: they become clinical events. That is exactly
        why they are not separate branches.
        """
        return bool(self.clinical_events or self.speech_recordings or self.documents)

    def has_ultrasound(self) -> bool:
        return bool(self.ultrasound_inputs)

    def has_temporal(self) -> bool:
        return self.temporal_series is not None and self.temporal_series.n_days > 0

    def declared_modalities(self) -> list[str]:
        """Modalities this bundle can drive, in canonical token naming."""
        present: list[str] = []
        if self.has_static_input():
            present.append("static_clinical")
        if self.has_ultrasound():
            present.append("ovarian_ultrasound")
        if self.has_temporal():
            present.append("longitudinal_hormonal_state")
        return present
