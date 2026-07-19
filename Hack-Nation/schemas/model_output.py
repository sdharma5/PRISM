"""Experiment-level result contracts shared by every training script."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class FoldMetrics(BaseModel):
    fold: int
    seed: int
    n_train: int
    n_test: int
    metrics: dict[str, float] = Field(default_factory=dict)


class CalibrationMetrics(BaseModel):
    brier: float | None = None
    ece: float | None = None
    calibration_slope: float | None = None
    calibration_intercept: float | None = None
    n_bins: int = 10


class ExperimentResult(BaseModel):
    """The artifact every experiment writes to ``metrics.json``."""

    experiment_id: str
    dataset_version: str
    git_commit: str
    model: str
    target: str
    split_strategy: str
    seeds: list[int] = Field(default_factory=list)
    fold_metrics: list[FoldMetrics] = Field(default_factory=list)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    aggregate_metrics_std: dict[str, float] = Field(default_factory=dict)
    calibration_metrics: CalibrationMetrics = Field(default_factory=CalibrationMetrics)
    feature_manifest: str | None = None
    split_manifest: str | None = None
    limitations: list[str] = Field(default_factory=list)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path


class ModelCardMetadata(BaseModel):
    """What ``BasePrismModel.export_model_card_metadata`` must return."""

    model_name: str
    model_version: str
    intended_use: str
    out_of_scope_uses: list[str] = Field(default_factory=list)
    training_datasets: list[str] = Field(default_factory=list)
    evaluation_datasets: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    ethical_considerations: list[str] = Field(default_factory=list)
    non_diagnostic_statement: str = (
        "This model is a research artifact. It does not diagnose any condition, "
        "does not provide medical advice, and is not validated for clinical use."
    )
