"""Patient-level containers. Deliberately thin: PRISM is event-centric."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.event import HormonalHealthEvent


class PatientRecord(BaseModel):
    """All events known for one patient within one dataset context.

    ``patient_id`` is scoped by ``source_dataset``. Identifiers from different
    datasets must never be merged — they describe different people.
    """

    patient_id: str
    source_dataset: str | None = None
    events: list[HormonalHealthEvent] = Field(default_factory=list)

    def by_code(self, code: str) -> list[HormonalHealthEvent]:
        return [e for e in self.events if e.canonical_variable_code == code]

    def model_ready_events(self) -> list[HormonalHealthEvent]:
        return [e for e in self.events if e.is_model_ready]


class SplitManifest(BaseModel):
    """Saved patient-ID split. Splits are always by patient, never by row."""

    manifest_id: str
    dataset_id: str
    dataset_version: str
    strategy: Literal[
        "repeated_stratified_kfold",
        "grouped_kfold",
        "leave_one_participant_out",
        "holdout",
    ]
    n_splits: int
    seeds: list[int] = Field(default_factory=list)
    folds: list[dict[str, list[str]]] = Field(default_factory=list)
    holdout_ids: list[str] = Field(default_factory=list)

    def assert_disjoint(self) -> None:
        """Fail loudly if any patient appears in both train and test of a fold."""
        for i, fold in enumerate(self.folds):
            overlap = set(fold.get("train", [])) & set(fold.get("test", []))
            if overlap:
                raise ValueError(f"Fold {i} leaks {len(overlap)} patient(s) across train/test.")
            if self.holdout_ids:
                bleed = set(self.holdout_ids) & (
                    set(fold.get("train", [])) | set(fold.get("test", []))
                )
                if bleed:
                    raise ValueError(f"Fold {i} overlaps the untouched holdout set.")
