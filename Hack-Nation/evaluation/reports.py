"""Assembling fold results into an ``ExperimentResult`` and a readable README.

Every experiment ships its limitations alongside its numbers. A metrics file
without them invites exactly the over-reading this project is built to avoid.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.calibration import calibration_report
from evaluation.classification import aggregate_fold_metrics
from schemas.model_output import ExperimentResult, FoldMetrics

#: Limitations attached to every experiment unless the caller replaces them.
DEFAULT_LIMITATIONS: tuple[str, ...] = (
    "Research artifact only: these outputs are phenotype profiles, not diagnoses, "
    "and carry no clinical decision authority.",
    "Labels are dataset-provided and were not re-adjudicated against current criteria.",
    "Missingness is informative; part of any apparent signal reflects who was measured.",
    "Cross-validated estimates on a single cohort are optimistic relative to external "
    "validation, which has not been performed.",
    "Cohort demographics are not representative; subgroup metrics must be read before "
    "generalizing any number here.",
)


def build_experiment_result(
    *,
    experiment_id: str,
    dataset_version: str,
    git_commit: str,
    model: str,
    target: str,
    split_strategy: str,
    seeds: Sequence[int],
    fold_records: Sequence[dict[str, Any]],
    predictions: pd.DataFrame | None = None,
    feature_manifest: str | None = None,
    split_manifest: str | None = None,
    limitations: Sequence[str] | None = None,
    n_bins: int = 10,
) -> ExperimentResult:
    """Aggregate per-fold records into the experiment-level result contract.

    ``fold_records`` entries carry ``fold``, ``seed``, ``n_train``, ``n_test`` and
    a ``metrics`` dict.
    """
    fold_metrics = [
        FoldMetrics(
            fold=int(record["fold"]),
            seed=int(record.get("seed", 0)),
            n_train=int(record.get("n_train", 0)),
            n_test=int(record.get("n_test", 0)),
            metrics={
                k: float(v)
                for k, v in record.get("metrics", {}).items()
                if isinstance(v, int | float | np.floating)
            },
        )
        for record in fold_records
    ]

    mean, std = aggregate_fold_metrics([fm.metrics for fm in fold_metrics])

    # Pooled calibration across all out-of-fold predictions: per-fold calibration
    # slopes are unstable on small folds, pooled ones are not.
    calibration = calibration_report([], [], n_bins=n_bins)
    if predictions is not None and {"y_true", "y_prob"}.issubset(predictions.columns):
        calibration = calibration_report(
            predictions["y_true"].to_numpy(), predictions["y_prob"].to_numpy(), n_bins=n_bins
        )

    return ExperimentResult(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        model=model,
        target=target,
        split_strategy=split_strategy,
        seeds=[int(s) for s in seeds],
        fold_metrics=fold_metrics,
        aggregate_metrics={k: v for k, v in mean.items() if np.isfinite(v)},
        aggregate_metrics_std={k: v for k, v in std.items() if np.isfinite(v)},
        calibration_metrics=calibration,
        feature_manifest=feature_manifest,
        split_manifest=split_manifest,
        limitations=list(limitations or DEFAULT_LIMITATIONS),
    )


def format_metric_table(
    mean: dict[str, float],
    std: dict[str, float],
    *,
    keys: Sequence[str] | None = None,
) -> str:
    """Markdown table of ``mean ± std`` for the headline metrics."""
    keys = keys or [
        "auroc",
        "auprc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "brier",
        "ece",
    ]
    lines = ["| metric | mean | std |", "| --- | --- | --- |"]
    for key in keys:
        if key not in mean or not np.isfinite(mean[key]):
            continue
        lines.append(f"| {key} | {mean[key]:.3f} | {std.get(key, float('nan')):.3f} |")
    return "\n".join(lines)


def write_experiment_readme(
    path: Path,
    result: ExperimentResult,
    *,
    config_summary: dict[str, Any] | None = None,
    subgroup_summary: dict[str, Any] | None = None,
    extra_sections: dict[str, str] | None = None,
) -> Path:
    """Write the human-readable summary that sits beside ``metrics.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        f"# Experiment: {result.experiment_id}",
        "",
        "> Research artifact. These outputs describe phenotype profiles on a research "
        "cohort. They are not a diagnosis, not medical advice, and not validated for "
        "clinical use.",
        "",
        "## Setup",
        "",
        f"- Model: `{result.model}`",
        f"- Target: `{result.target}`",
        f"- Split strategy: `{result.split_strategy}`",
        f"- Seeds: {result.seeds}",
        f"- Folds: {len(result.fold_metrics)}",
        f"- Dataset version: `{result.dataset_version}`",
        f"- Git commit: `{result.git_commit}`",
        "",
        "## Results (mean across folds)",
        "",
        format_metric_table(result.aggregate_metrics, result.aggregate_metrics_std),
        "",
        "## Calibration (pooled out-of-fold)",
        "",
        f"- Brier: {_fmt(result.calibration_metrics.brier)}",
        f"- ECE ({result.calibration_metrics.n_bins} bins): {_fmt(result.calibration_metrics.ece)}",
        f"- Calibration slope: {_fmt(result.calibration_metrics.calibration_slope)} (1.0 is ideal)",
        f"- Calibration intercept: {_fmt(result.calibration_metrics.calibration_intercept)} "
        "(0.0 is ideal)",
        "",
    ]

    if subgroup_summary:
        sections += ["## Subgroup performance", ""]
        for column, payload in subgroup_summary.items():
            gaps = payload.get("gaps", {}) if isinstance(payload, dict) else {}
            summary = ", ".join(f"{k}={_fmt(v)}" for k, v in gaps.items())
            sections.append(f"- `{column}`: {summary}")
        sections.append("")

    if config_summary:
        sections += ["## Resolved configuration (excerpt)", "", "```yaml"]
        sections += [f"{k}: {v}" for k, v in config_summary.items()]
        sections += ["```", ""]

    for title, body in (extra_sections or {}).items():
        sections += [f"## {title}", "", body, ""]

    sections += ["## Limitations", ""]
    sections += [f"- {item}" for item in result.limitations]
    sections += [
        "",
        "## Artifacts in this directory",
        "",
        "- `config.resolved.yaml` — the fully resolved configuration this run used",
        "- `environment.json` — python and package versions",
        "- `git_commit.txt` — repository state",
        "- `split_manifest.json` — patient-level folds (disjointness enforced)",
        "- `feature_manifest.json` — exact columns, transforms and statistics",
        "- `metrics.json` — the `ExperimentResult`",
        "- `predictions.csv` — out-of-fold predictions",
        "",
    ]

    path.write_text("\n".join(sections))
    return path


def _fmt(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def write_predictions(predictions: pd.DataFrame, path: Path) -> Path:
    """Write out-of-fold predictions, preferring parquet when an engine exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            predictions.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
    predictions.to_csv(path, index=False)
    return path


__all__ = [
    "DEFAULT_LIMITATIONS",
    "build_experiment_result",
    "format_metric_table",
    "write_experiment_readme",
    "write_predictions",
]
