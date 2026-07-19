"""Universal hormonal-health event.

This is the single structured record type that every ingestion adapter emits and
that the event store persists. It is deliberately condition-agnostic: nothing in
this module may reference PMOS-specific fields. PMOS logic belongs in
``models/adapters/pmos/``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0.0"

Modality = Literal[
    "questionnaire",
    "patient_voice",
    "clinician_voice",
    "laboratory",
    "clinical_document",
    "ultrasound_report",
    "ultrasound_image",
    "wearable",
    "cgm",
    "menstrual_history",
    "medication",
    "diagnosis_history",
]

Provenance = Literal[
    "patient_confirmed",
    "clinician_confirmed",
    "document_extracted",
    "device_measured",
    "dataset_provided",
    "model_measured",
    "model_inferred",
]

ConfirmationStatus = Literal[
    "confirmed",
    "awaiting_patient_confirmation",
    "awaiting_clinician_confirmation",
    "rejected",
    "not_required",
]

MissingnessStatus = Literal[
    "observed",
    "not_collected",
    "not_available",
    "not_applicable",
    "extraction_failed",
    "intentionally_masked",
]

#: Provenance values whose events may never be self-declared as ``confirmed``.
UNREVIEWED_PROVENANCE: frozenset[str] = frozenset(
    {"document_extracted", "model_measured", "model_inferred"}
)

#: Modalities that must carry an evidence span or a source location.
EVIDENCE_REQUIRED_MODALITIES: frozenset[str] = frozenset(
    {"patient_voice", "clinician_voice", "clinical_document", "ultrasound_report"}
)


class HormonalHealthEvent(BaseModel):
    """One observation about one patient, with full provenance.

    Raw source values are never mutated in place: ``value`` holds the canonical
    value and ``raw_value`` / ``raw_unit`` preserve exactly what the source said.
    """

    event_id: UUID = Field(default_factory=uuid4)

    patient_id: str
    variable_name: str
    canonical_variable_code: str

    value: Any = None
    unit: str | None = None

    # Immutable record of the source value, retained even after normalization.
    raw_value: Any = None
    raw_unit: str | None = None

    observed_at: datetime | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None

    modality: Modality
    provenance: Provenance

    extraction_confidence: float = Field(ge=0.0, le=1.0)
    confirmation_status: ConfirmationStatus
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    missingness_status: MissingnessStatus = "observed"

    negated: bool = False
    historical: bool = False
    uncertain: bool = False

    source_dataset: str | None = None
    source_file_id: str | None = None
    source_file_hash: str | None = None
    source_page: int | None = None
    source_time_start_seconds: float | None = None
    source_time_end_seconds: float | None = None
    evidence_text: str | None = None

    parser_version: str | None = None
    model_version: str | None = None
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_contracts(self) -> HormonalHealthEvent:
        if self.missingness_status == "observed" and self.value is None:
            raise ValueError(
                f"{self.canonical_variable_code}: missingness_status='observed' requires a value; "
                "use an explicit missingness status instead of a silent null."
            )
        if self.missingness_status != "observed" and self.value is not None:
            raise ValueError(
                f"{self.canonical_variable_code}: a non-observed event must not carry a value."
            )

        if (
            self.modality == "laboratory"
            and self.missingness_status == "observed"
            and self.unit is None
            and self.raw_unit is None
        ):
            raise ValueError(
                f"{self.canonical_variable_code}: laboratory events require a unit; set "
                "raw_unit='dimensionless' if the source genuinely has none."
            )

        if (
            self.provenance in UNREVIEWED_PROVENANCE
            and self.confirmation_status == "confirmed"
            and self.reviewed_by is None
        ):
            raise ValueError(
                f"{self.canonical_variable_code}: provenance='{self.provenance}' cannot be "
                "'confirmed' without reviewed_by — human review is required."
            )

        if self.modality in EVIDENCE_REQUIRED_MODALITIES and self.missingness_status == "observed":
            has_location = (
                self.evidence_text is not None
                or self.source_page is not None
                or self.source_time_start_seconds is not None
            )
            if not has_location:
                raise ValueError(
                    f"{self.canonical_variable_code}: modality='{self.modality}' requires "
                    "evidence_text or a source location."
                )

        if self.negated and self.uncertain and self.confirmation_status == "confirmed":
            raise ValueError("An event cannot be simultaneously confirmed, negated and uncertain.")

        return self

    # -- Semantics helpers -------------------------------------------------

    @property
    def is_model_ready(self) -> bool:
        """True when this event may enter a model-ready snapshot."""
        return self.missingness_status == "observed" and self.confirmation_status in {
            "confirmed",
            "not_required",
        }

    @property
    def asserts_presence(self) -> bool:
        """True when the event asserts a currently present, non-negated finding."""
        return not self.negated and not self.historical and self.missingness_status == "observed"
