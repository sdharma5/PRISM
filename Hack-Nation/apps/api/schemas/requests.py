"""Request bodies for the patient inference API.

Missing means absent, not zero. Optional fields default to ``None`` or empty,
and nothing invents a value the caller didn't send. Sending ``0`` for an
unmeasured fasting glucose is a different clinical claim, and nothing downstream
can tell the two apart once it's in a token.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inference.patient_bundle import PatientDataBundle, TemporalInput
from schemas.event import HormonalHealthEvent
from schemas.temporal import ParticipantDay

__all__ = [
    "PatientInferenceRequest",
    "StaticInferenceRequest",
    "TemporalInferenceRequest",
    "UltrasoundInferenceRequest",
]

#: Scalar types a clinical feature may take. `None` is meaningful and preserved.
ClinicalValue = float | int | str | bool | None


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatientInferenceRequest(_Base):
    """The main inference request.

    Every modality is optional; the service runs what it has inputs for and
    reports the rest as missing.
    """

    patient_id: str = Field(min_length=1)

    clinical_features: dict[str, ClinicalValue] | None = Field(
        default=None,
        description=(
            "Canonical variable code -> value, for values entered directly rather "
            "than arriving as confirmed events. Codes must exist in "
            "registry/variables.yaml; unknown codes are dropped at the encoder "
            "boundary rather than treated as evidence."
        ),
    )
    confirmed_events: list[HormonalHealthEvent] = Field(
        default_factory=list,
        description="Events the patient or a clinician has confirmed.",
    )
    temporal_observations: list[ParticipantDay] = Field(
        default_factory=list,
        description=(
            "Daily longitudinal observations. Named `temporal_observations` for "
            "the API surface; the internal type is ParticipantDay."
        ),
    )
    ultrasound_job_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Completed ultrasound job identifiers. Accepted so the contract is "
            "stable, but the ultrasound branch is gated off for inference; see "
            "GET /api/v1/models/status."
        ),
    )
    requested_adapter: Literal["pmos"] = "pmos"

    @field_validator("clinical_features")
    @classmethod
    def _reject_all_null_features(
        cls, value: dict[str, ClinicalValue] | None
    ) -> dict[str, ClinicalValue] | None:
        """An all-null feature map usually means a form template shipped without
        values. Rejecting it beats returning a confident-looking abstention."""
        if value is not None and value and all(v is None for v in value.values()):
            raise ValueError(
                "clinical_features contained only null values. Omit the field entirely "
                "rather than sending placeholder nulls: missing means absent."
            )
        return value

    def to_bundle(self) -> PatientDataBundle:
        """Convert to the internal bundle the orchestrator consumes.

        Raises:
            ValueError: If any nested record names a different patient. The
                bundle enforces this itself; the router maps it to a 422.
        """
        events = list(self.confirmed_events)
        events.extend(_events_from_features(self.clinical_features, self.patient_id))

        temporal = (
            TemporalInput(participant_days=list(self.temporal_observations))
            if self.temporal_observations
            else None
        )

        return PatientDataBundle(
            patient_id=self.patient_id,
            clinical_events=events,
            temporal_series=temporal,
        )


class StaticInferenceRequest(_Base):
    """Static-clinical branch only."""

    patient_id: str = Field(min_length=1)
    clinical_features: dict[str, ClinicalValue] | None = None
    confirmed_events: list[HormonalHealthEvent] = Field(default_factory=list)

    def to_bundle(self) -> PatientDataBundle:
        events = list(self.confirmed_events)
        events.extend(_events_from_features(self.clinical_features, self.patient_id))
        return PatientDataBundle(patient_id=self.patient_id, clinical_events=events)


class TemporalInferenceRequest(_Base):
    """Longitudinal branch only. Cannot produce a whole-patient PMOS score."""

    patient_id: str = Field(min_length=1)
    temporal_observations: list[ParticipantDay] = Field(default_factory=list, min_length=1)

    def to_bundle(self) -> PatientDataBundle:
        return PatientDataBundle(
            patient_id=self.patient_id,
            temporal_series=TemporalInput(participant_days=list(self.temporal_observations)),
        )


class UltrasoundInferenceRequest(_Base):
    """Ultrasound branch only. Gated off; see the router for the 503 it returns."""

    patient_id: str = Field(min_length=1)
    job_ids: list[str] = Field(default_factory=list)


def _events_from_features(
    features: dict[str, ClinicalValue] | None, patient_id: str
) -> list[HormonalHealthEvent]:
    """Flat feature map -> events, skipping nulls.

    An event asserts an observation happened; a null is the absence of one.
    Provenance is ``patient_confirmed`` since this path is for directly-entered
    values -- device measurements and document extractions should arrive as
    proper events carrying their own provenance.
    """
    if not features:
        return []

    events: list[HormonalHealthEvent] = []
    for code, value in features.items():
        if value is None:
            continue
        events.append(
            HormonalHealthEvent(
                patient_id=patient_id,
                variable_name=code,
                canonical_variable_code=code,
                value=value,
                modality="questionnaire",
                provenance="patient_confirmed",
                extraction_confidence=1.0,
                confirmation_status="confirmed",
            )
        )
    return events
