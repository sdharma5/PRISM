"""Subgroup metrics.

An aggregate number can look healthy while the model fails a subgroup entirely.
Subgroup breakdowns are computed by default, with explicit small-n suppression so
nobody quotes an AUROC computed on nine people.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from evaluation.calibration import calibration_metrics_dict
from evaluation.classification import classification_metrics

#: Below this many patients a subgroup metric is reported but flagged unreliable.
MIN_RELIABLE_SUBGROUP_N = 30


def subgroup_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    groups: Sequence[object],
    *,
    threshold: float = 0.5,
    min_n: int = MIN_RELIABLE_SUBGROUP_N,
    include_calibration: bool = True,
) -> pd.DataFrame:
    """One row of metrics per subgroup, with an ``is_reliable`` flag."""
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=float),
            "y_score": np.asarray(y_score, dtype=float),
            "group": [str(g) for g in groups],
        }
    )

    rows: list[dict[str, object]] = []
    for name, chunk in frame.groupby("group", sort=True):
        metrics = classification_metrics(
            chunk["y_true"].to_numpy(), chunk["y_score"].to_numpy(), threshold=threshold
        )
        if include_calibration:
            metrics.update(
                calibration_metrics_dict(chunk["y_true"].to_numpy(), chunk["y_score"].to_numpy())
            )
        row: dict[str, object] = {"group": name, "n": int(len(chunk))}
        row.update({k: float(v) for k, v in metrics.items()})
        row["is_reliable"] = bool(len(chunk) >= min_n and chunk["y_true"].nunique() > 1)
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("group").reset_index(drop=True)
    return result


def subgroup_gaps(table: pd.DataFrame, metric: str = "auroc") -> dict[str, float]:
    """Worst-group value and the max-min gap, restricted to reliable subgroups.

    The gap is the number that matters for equity claims; the mean across groups
    can hide it entirely.
    """
    if table.empty or metric not in table.columns:
        return {}
    reliable = table
    if "is_reliable" in table:
        reliable = table[table["is_reliable"].astype(bool)]
    values = pd.to_numeric(reliable[metric], errors="coerce").dropna()
    if values.empty:
        return {"n_reliable_groups": 0.0}
    return {
        "n_reliable_groups": float(len(values)),
        f"{metric}_min": float(values.min()),
        f"{metric}_max": float(values.max()),
        f"{metric}_gap": float(values.max() - values.min()),
        f"{metric}_mean": float(values.mean()),
    }


def subgroup_report(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    frame: pd.DataFrame,
    grouping_columns: Sequence[str],
    *,
    threshold: float = 0.5,
    min_n: int = MIN_RELIABLE_SUBGROUP_N,
) -> dict[str, object]:
    """Subgroup tables and gaps for several grouping columns at once."""
    report: dict[str, object] = {}
    for column in grouping_columns:
        if column not in frame.columns:
            report[column] = {"error": f"grouping column '{column}' not present"}
            continue
        table = subgroup_metrics(
            y_true, y_score, frame[column].to_numpy(), threshold=threshold, min_n=min_n
        )
        report[column] = {
            "table": table.to_dict(orient="records"),
            "gaps": subgroup_gaps(table, "auroc"),
        }
    return report


def bin_continuous_group(
    values: Sequence[float] | np.ndarray,
    *,
    edges: Sequence[float] = (0, 25, 35, 45, 200),
    label_prefix: str = "",
) -> list[str]:
    """Bucket a continuous variable (e.g. age, BMI) into labelled subgroups."""
    arr = np.asarray(values, dtype=float)
    labels = [f"{label_prefix}{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
    idx = np.clip(np.digitize(arr, np.asarray(edges[1:-1], dtype=float)), 0, len(labels) - 1)
    return [labels[i] if np.isfinite(v) else "unknown" for i, v in zip(idx, arr, strict=True)]


__all__ = [
    "MIN_RELIABLE_SUBGROUP_N",
    "bin_continuous_group",
    "subgroup_gaps",
    "subgroup_metrics",
    "subgroup_report",
]
