"""The feature manifest: exactly which columns, transforms and statistics were used.

Metrics are only interpretable if you can reconstruct the matrix that produced
them. This manifest is written next to every experiment's metrics so a reviewer
can tell whether a score came from labs, from symptoms, or mostly from
missingness indicators.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from features.static_features import (
    StaticFeatureMatrix,
    indicator_columns_of,
    value_columns_of,
)


class ColumnStatistics(BaseModel):
    """Descriptive statistics of one feature column, computed on the fitting data."""

    name: str
    group: str | None = None
    dtype: str = "float64"
    n_observed: int = 0
    n_missing: int = 0
    observed_fraction: float = 0.0
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    n_unique: int = 0


class FeatureManifest(BaseModel):
    """A reproducible description of one assembled feature matrix."""

    manifest_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    dataset_id: str | None = None
    dataset_version: str | None = None
    n_rows: int = 0
    n_features: int = 0
    label_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    value_columns: list[str] = Field(default_factory=list)
    indicator_columns: list[str] = Field(default_factory=list)
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    derived_columns: list[str] = Field(default_factory=list)
    derivations: dict[str, str] = Field(default_factory=dict)
    transforms: list[dict[str, Any]] = Field(default_factory=list)
    column_statistics: list[ColumnStatistics] = Field(default_factory=list)
    missingness_summary: dict[str, dict[str, int]] = Field(default_factory=dict)
    dropped_columns: list[str] = Field(default_factory=list)
    domain_scoring: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path

    @classmethod
    def read_json(cls, path: Path) -> FeatureManifest:
        return cls.model_validate(json.loads(Path(path).read_text()))


def _column_statistics(X: pd.DataFrame, groups: dict[str, list[str]]) -> list[ColumnStatistics]:
    lookup = {col: group for group, cols in groups.items() for col in cols}
    stats: list[ColumnStatistics] = []
    for col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        observed = series.dropna()
        stats.append(
            ColumnStatistics(
                name=str(col),
                group=lookup.get(col),
                dtype=str(X[col].dtype),
                n_observed=int(observed.size),
                n_missing=int(series.isna().sum()),
                observed_fraction=float(observed.size / len(series)) if len(series) else 0.0,
                mean=float(observed.mean()) if observed.size else None,
                std=float(observed.std(ddof=0)) if observed.size > 1 else None,
                minimum=float(observed.min()) if observed.size else None,
                maximum=float(observed.max()) if observed.size else None,
                n_unique=int(observed.nunique()),
            )
        )
    return stats


def build_feature_manifest(
    matrix: StaticFeatureMatrix,
    *,
    manifest_id: str,
    dataset_id: str | None = None,
    dataset_version: str | None = None,
    label_column: str | None = "pmos_binary",
    transforms: list[dict[str, Any]] | None = None,
    domain_scoring: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> FeatureManifest:
    """Build the manifest from an assembled :class:`StaticFeatureMatrix`."""
    X = matrix.X
    return FeatureManifest(
        manifest_id=manifest_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        n_rows=int(len(X)),
        n_features=int(X.shape[1]),
        label_column=label_column,
        feature_columns=[str(c) for c in X.columns],
        value_columns=[str(c) for c in value_columns_of(X)],
        indicator_columns=[str(c) for c in indicator_columns_of(X)],
        feature_groups={g: [str(c) for c in cols] for g, cols in matrix.feature_groups.items()},
        derived_columns=list(matrix.derived_columns),
        derivations=dict(matrix.derivation_notes),
        transforms=list(transforms or []),
        column_statistics=_column_statistics(X, matrix.feature_groups),
        missingness_summary=matrix.missingness_summary,
        dropped_columns=list(matrix.dropped_columns),
        domain_scoring=dict(domain_scoring or {}),
        notes=list(notes or []),
    )


def describe_pipeline(pipeline: Any) -> list[dict[str, Any]]:
    """Summarize a fitted sklearn pipeline into manifest-friendly transform records.

    Recorded per step, including the fitted statistics themselves, so a reader can
    verify that (for example) the imputer used training-fold medians.
    """
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        return [{"step": "unknown", "class": type(pipeline).__name__}]

    records: list[dict[str, Any]] = []
    for name, step in steps:
        record: dict[str, Any] = {"step": name, "class": type(step).__name__}
        for attr in ("strategy", "add_indicator", "with_mean", "with_std"):
            if hasattr(step, attr):
                record[attr] = getattr(step, attr)
        for attr, key in (
            ("statistics_", "fitted_statistics"),
            ("mean_", "fitted_mean"),
            ("scale_", "fitted_scale"),
        ):
            value = getattr(step, attr, None)
            if value is not None:
                record[key] = np.asarray(value, dtype=float).round(6).tolist()
        records.append(record)
    return records


__all__ = [
    "ColumnStatistics",
    "FeatureManifest",
    "build_feature_manifest",
    "describe_pipeline",
]
