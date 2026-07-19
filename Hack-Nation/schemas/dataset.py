"""Dataset registry, variable registry, and processing-manifest contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

VariableType = Literal[
    "continuous", "integer", "binary", "categorical", "ordinal", "datetime", "text"
]


class ValidRange(BaseModel):
    min: float | None = None
    max: float | None = None

    def contains(self, value: float) -> bool:
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)


class VariableSpec(BaseModel):
    """One canonical variable. Source columns map onto these, never vice versa."""

    canonical_name: str
    type: VariableType
    unit: str | None = None
    canonical_unit: str | None = None
    domain: list[str] = Field(default_factory=list)
    valid_range: ValidRange | None = None
    categories: list[str] | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> VariableSpec:
        if self.type == "categorical" and not self.categories:
            raise ValueError(f"{self.canonical_name}: categorical variables must list categories.")
        return self


class DatasetSpec(BaseModel):
    """What a dataset is, what it may support, and what it may never claim."""

    name: str
    modality: list[str]
    access: str = "unspecified"
    patient_level: bool = True
    longitudinal: bool = False
    labels: list[str] = Field(default_factory=list)
    allowed_uses: list[str] = Field(default_factory=list)
    prohibited_claims: list[str] = Field(default_factory=list)
    citation: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> DatasetSpec:
        if not self.allowed_uses:
            raise ValueError(f"{self.name}: every dataset must declare allowed_uses.")
        overlap = set(self.allowed_uses) & set(self.prohibited_claims)
        if overlap:
            raise ValueError(f"{self.name}: {sorted(overlap)} is both allowed and prohibited.")
        return self

    def permits(self, use: str) -> bool:
        return use in self.allowed_uses


class DatasetRegistry(BaseModel):
    datasets: dict[str, DatasetSpec]

    def require(self, dataset_id: str, use: str) -> DatasetSpec:
        """Fetch a dataset spec, refusing a use the registry does not allow."""
        spec = self.datasets.get(dataset_id)
        if spec is None:
            raise KeyError(
                f"Unknown dataset '{dataset_id}'. Register it in registry/datasets.yaml."
            )
        if not spec.permits(use):
            raise PermissionError(
                f"Dataset '{dataset_id}' does not allow use '{use}'. "
                f"Allowed: {sorted(spec.allowed_uses)}."
            )
        return spec


class VariableRegistry(BaseModel):
    variables: dict[str, VariableSpec]

    def get(self, code: str) -> VariableSpec:
        if code not in self.variables:
            raise KeyError(
                f"Unknown canonical variable '{code}'. Add it to registry/variables.yaml."
            )
        return self.variables[code]


class DroppedRecord(BaseModel):
    record_index: int | str
    reason: str
    detail: str = ""


class ProcessingManifest(BaseModel):
    """Written to ``artifacts/manifests/<dataset>/<version>/``."""

    dataset_id: str
    dataset_version: str
    adapter: str
    adapter_version: str
    created_at: str
    file_checksums: dict[str, str] = Field(default_factory=dict)
    variable_mapping: dict[str, str] = Field(default_factory=dict)
    excluded_source_columns: dict[str, str] = Field(default_factory=dict)
    n_source_records: int = 0
    n_events_emitted: int = 0
    n_dropped: int = 0
    unit_conversions_applied: dict[str, int] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
