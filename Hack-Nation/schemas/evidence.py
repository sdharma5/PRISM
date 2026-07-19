"""Evidence-level structures: conflicts, snapshots, and confirmation batches."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from schemas.event import ConfirmationStatus, HormonalHealthEvent, MissingnessStatus

ConflictType = Literal[
    "value_disagreement",
    "unit_disagreement",
    "temporal_disagreement",
    "presence_vs_negation",
    "duplicate_measurement",
]


class EvidenceConflict(BaseModel):
    """Two or more events disagree about the same variable.

    PRISM never overwrites the loser. Both events stay in the store and the
    conflict is surfaced to a human.
    """

    variable_name: str
    canonical_variable_code: str
    event_ids: list[str]
    conflict_type: ConflictType
    detail: str = ""
    recommended_resolution: str
    requires_human_review: bool = True


class SnapshotValue(BaseModel):
    """One selected value in a model-ready snapshot, with its receipt."""

    canonical_variable_code: str
    value: Any = None
    unit: str | None = None
    missingness_status: MissingnessStatus
    observed_at: datetime | None = None
    recency_days: float | None = None
    source_event_id: str | None = None
    provenance: str | None = None
    confirmation_status: ConfirmationStatus | None = None
    quality: float | None = None
    n_candidates: int = 0


class PatientSnapshot(BaseModel):
    """A point-in-time, model-ready view assembled from the event store."""

    patient_id: str
    as_of: datetime
    created_at: datetime
    values: dict[str, SnapshotValue] = Field(default_factory=dict)
    missingness_mask: dict[str, bool] = Field(default_factory=dict)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    excluded_event_ids: list[str] = Field(default_factory=list)
    exclusion_reasons: dict[str, str] = Field(default_factory=dict)
    included_modalities: list[str] = Field(default_factory=list)
    allowed_confirmation_statuses: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def observed_codes(self) -> list[str]:
        return sorted(c for c, v in self.values.items() if v.missingness_status == "observed")

    def coverage(self, expected_codes: list[str]) -> float:
        if not expected_codes:
            return 0.0
        observed = set(self.observed_codes())
        return sum(1 for c in expected_codes if c in observed) / len(expected_codes)


class ConfirmationBatch(BaseModel):
    """What a review UI receives and returns. Only ``confirmed`` reaches models."""

    patient_id: str
    confirmed: list[HormonalHealthEvent] = Field(default_factory=list)
    awaiting_confirmation: list[HormonalHealthEvent] = Field(default_factory=list)
    rejected: list[HormonalHealthEvent] = Field(default_factory=list)
