"""Binary-classification metrics, reported as a group rather than one headline number.

AUROC alone hides the two failure modes that matter most in an imbalanced
hormonal-health cohort: poor precision on the positive class, and miscalibrated
probabilities. Every caller gets the whole panel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

#: The metric keys every classification evaluation emits.
CLASSIFICATION_METRICS: tuple[str, ...] = (
    "auroc",
    "auprc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "precision",
    "npv",
    "accuracy",
    "mcc",
    "prevalence",
)


def _clean(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either the label or the score is missing."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_score {y_score_arr.shape}."
        )
    keep = np.isfinite(y_true_arr) & np.isfinite(y_score_arr)
    return y_true_arr[keep], y_score_arr[keep]


def threshold_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Sensitivity/specificity/F1 and friends at an already-applied threshold."""
    y_true_arr, y_pred_arr = _clean(y_true, y_pred)
    if y_true_arr.size == 0:
        return {}

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(
        y_true_arr.astype(int), y_pred_arr.astype(int), labels=labels
    ).ravel()

    def _safe(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else float("nan")

    return {
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "precision": _safe(tp, tp + fp),
        "npv": _safe(tn, tn + fn),
        "accuracy": _safe(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_arr.astype(int), y_pred_arr.astype(int))
        ),
        "f1": float(f1_score(y_true_arr.astype(int), y_pred_arr.astype(int), zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr.astype(int), y_pred_arr.astype(int))),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def classification_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Full metric panel from continuous scores plus a decision threshold."""
    y_true_arr, y_score_arr = _clean(y_true, y_score)
    if y_true_arr.size == 0:
        return dict.fromkeys(CLASSIFICATION_METRICS, float("nan"))

    metrics: dict[str, float] = {"n": float(y_true_arr.size)}
    metrics["prevalence"] = float(np.mean(y_true_arr))

    # AUROC/AUPRC are undefined with a single observed class — report NaN, never 0.5.
    if len(np.unique(y_true_arr)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        metrics["auprc"] = float(average_precision_score(y_true_arr, y_score_arr))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    metrics.update(threshold_metrics(y_true_arr, (y_score_arr >= threshold).astype(int)))
    return metrics


#: Objectives :func:`sweep_thresholds` can evaluate without a per-threshold loop.
SWEEPABLE_OBJECTIVES: frozenset[str] = frozenset(
    {"balanced_accuracy", "f1", "accuracy", "youden_j"}
)


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    candidates: np.ndarray,
    objective: str,
) -> np.ndarray:
    """Objective value at every candidate threshold, computed in one vectorized pass.

    Written as cumulative counts rather than a loop over confusion matrices: the
    single-feature baseline sweeps every column at every threshold, and the naive
    version costs minutes per fold.
    """
    order = np.argsort(y_score, kind="mergesort")
    scores_sorted = y_score[order]
    labels_sorted = y_true[order]

    cum_pos = np.concatenate([[0.0], np.cumsum(labels_sorted)])
    cum_all = np.arange(len(labels_sorted) + 1, dtype=float)

    # Rows at positions < idx fall below the threshold and are predicted negative.
    idx = np.searchsorted(scores_sorted, candidates, side="left")
    total_pos = float(labels_sorted.sum())
    total_neg = float(len(labels_sorted) - total_pos)

    fn = cum_pos[idx]
    tn = cum_all[idx] - fn
    tp = total_pos - fn
    fp = total_neg - tn

    with np.errstate(invalid="ignore", divide="ignore"):
        sensitivity = np.where(total_pos > 0, tp / max(total_pos, 1e-12), np.nan)
        specificity = np.where(total_neg > 0, tn / max(total_neg, 1e-12), np.nan)
        if objective == "balanced_accuracy":
            return (sensitivity + specificity) / 2.0
        if objective == "youden_j":
            return sensitivity + specificity - 1.0
        if objective == "accuracy":
            return (tp + tn) / float(len(labels_sorted))
        if objective == "f1":
            denominator = 2 * tp + fp + fn
            return np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1e-12), 0.0)
    raise ValueError(f"Objective '{objective}' cannot be swept vectorized.")


def best_threshold(
    y_true: Sequence[float] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    objective: str = "balanced_accuracy",
    max_candidates: int = 128,
) -> float:
    """Threshold maximizing an objective *on the data given*.

    Tune this on a training fold only: choosing the threshold on the test fold is
    a subtle form of leakage that quietly buys a few points of accuracy.
    """
    y_true_arr, y_score_arr = _clean(y_true, y_score)
    if y_true_arr.size == 0 or len(np.unique(y_true_arr)) < 2:
        return 0.5

    candidates = np.unique(y_score_arr)
    if candidates.size > max_candidates:
        # Quantile-spaced candidates: dense where the scores actually are.
        candidates = np.unique(np.quantile(y_score_arr, np.linspace(0.0, 1.0, max_candidates)))
    if candidates.size == 0:
        return 0.5

    if objective in SWEEPABLE_OBJECTIVES:
        values = sweep_thresholds(y_true_arr, y_score_arr, candidates, objective)
    else:
        values = np.asarray(
            [
                threshold_metrics(y_true_arr, (y_score_arr >= t).astype(int)).get(
                    objective, float("nan")
                )
                for t in candidates
            ],
            dtype=float,
        )

    if not np.isfinite(values).any():
        return 0.5
    return float(candidates[int(np.nanargmax(np.where(np.isfinite(values), values, -np.inf)))])


def aggregate_fold_metrics(
    fold_metrics: Sequence[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and standard deviation of each metric across folds.

    ``NaN`` values are ignored rather than propagated, so one degenerate fold does
    not erase an otherwise valid aggregate — but a metric that is NaN everywhere
    stays NaN.
    """
    keys: list[str] = []
    for fold in fold_metrics:
        for key in fold:
            if key not in keys:
                keys.append(key)

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(fold[key]) for fold in fold_metrics if key in fold], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            mean[key], std[key] = float("nan"), float("nan")
            continue
        mean[key] = float(values.mean())
        std[key] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return mean, std


__all__ = [
    "CLASSIFICATION_METRICS",
    "aggregate_fold_metrics",
    "best_threshold",
    "classification_metrics",
    "threshold_metrics",
]
