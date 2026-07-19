"""Calibration: is a predicted 0.7 actually 70%?

Discrimination and calibration are independent failures. A model can rank
patients perfectly (AUROC 0.9) while systematically over-stating risk, which is
exactly the failure that makes a research score dangerous if it is ever read as
a probability of having a condition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from schemas.model_output import CalibrationMetrics

_EPS = 1e-12


def _clean(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    keep = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
    return y_true_arr[keep], np.clip(y_prob_arr[keep], 0.0, 1.0)


def brier_score(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> float:
    """Mean squared error of the predicted probabilities."""
    y_true_arr, y_prob_arr = _clean(y_true, y_prob)
    if y_true_arr.size == 0:
        return float("nan")
    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def expected_calibration_error(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Binned |confidence - accuracy|, weighted by bin population."""
    y_true_arr, y_prob_arr = _clean(y_true, y_prob)
    if y_true_arr.size == 0:
        return float("nan")

    edges = _bin_edges(y_prob_arr, n_bins=n_bins, strategy=strategy)
    bins = np.clip(np.digitize(y_prob_arr, edges[1:-1], right=False), 0, len(edges) - 2)

    error, total = 0.0, float(y_true_arr.size)
    for b in range(len(edges) - 1):
        mask = bins == b
        if not mask.any():
            continue
        error += mask.sum() / total * abs(y_prob_arr[mask].mean() - y_true_arr[mask].mean())
    return float(error)


def _bin_edges(y_prob: np.ndarray, *, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(y_prob, quantiles))
        if edges.size < 2:
            return np.array([0.0, 1.0])
        return edges
    if strategy != "uniform":
        raise ValueError(f"Unknown binning strategy '{strategy}'.")
    return np.linspace(0.0, 1.0, n_bins + 1)


def calibration_curve_points(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> list[dict[str, float]]:
    """Per-bin (mean predicted, observed rate, count) — the reliability diagram data."""
    y_true_arr, y_prob_arr = _clean(y_true, y_prob)
    if y_true_arr.size == 0:
        return []

    edges = _bin_edges(y_prob_arr, n_bins=n_bins, strategy=strategy)
    bins = np.clip(np.digitize(y_prob_arr, edges[1:-1], right=False), 0, len(edges) - 2)

    points: list[dict[str, float]] = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if not mask.any():
            continue
        points.append(
            {
                "bin": float(b),
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_predicted": float(y_prob_arr[mask].mean()),
                "observed_rate": float(y_true_arr[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return points


def calibration_slope_intercept(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Slope and intercept of a logistic recalibration of ``logit(p)``.

    Perfect calibration is slope 1, intercept 0. Slope < 1 means predictions are
    too extreme; a negative intercept means systematic over-prediction.
    """
    y_true_arr, y_prob_arr = _clean(y_true, y_prob)
    if y_true_arr.size < 3 or len(np.unique(y_true_arr)) < 2:
        return float("nan"), float("nan")

    logits = _logit(y_prob_arr)
    if not np.isfinite(logits).all() or np.std(logits) < _EPS:
        return float("nan"), float("nan")

    # Effectively unpenalized (C -> inf): any regularization would shrink the
    # slope and make a miscalibrated model look better calibrated than it is.
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    model.fit(logits.reshape(-1, 1), y_true_arr.astype(int))
    return float(model.coef_[0][0]), float(model.intercept_[0])


def _logit(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(p, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def calibration_report(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationMetrics:
    """The full calibration block written into ``metrics.json``."""
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return CalibrationMetrics(
        brier=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy),
        calibration_slope=None if not np.isfinite(slope) else slope,
        calibration_intercept=None if not np.isfinite(intercept) else intercept,
        n_bins=n_bins,
    )


def calibration_metrics_dict(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    """Flat calibration metrics, for merging into a fold's metric dict."""
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


#: Number of equal-frequency reliability bins. Five, not ten: on a held-out set
#: of ~109 patients ten bins leaves ~11 patients each, and an observed rate over
#: 11 patients has a confidence interval wide enough to be consistent with almost
#: any claim. Five bins is the coarsest reading that still shows a monotone trend.
DEFAULT_N_BINS = 5

#: Below this, a bin's observed rate is reported but must not be reasoned from.
#: A 95% Wilson interval on 20 patients spans roughly +/-0.20 even at the extremes.
MIN_INTERPRETABLE_BIN_COUNT = 20

#: The only data a calibrator may be fitted on. Held-out predictions are for
#: measuring calibration, never for fitting it -- a calibrator fitted on the same
#: patients it is scored against reports its own training error as generalization.
ALLOWED_CALIBRATION_FIT_SOURCE = "train_out_of_fold"


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because reliability bins routinely
    contain observed rates at 0.0 or 1.0, where the normal interval has zero width
    and would report perfect certainty from a handful of patients.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def equal_frequency_bins(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> list[dict[str, Any]]:
    """Reliability table over ``n_bins`` bins of approximately equal population.

    Equal-frequency rather than equal-width: predicted probabilities on this
    cohort pile up at the extremes, so equal-width bins leave the middle nearly
    empty and invite conclusions from three patients. Equal-frequency guarantees
    every bin carries roughly the same evidential weight.

    Each row carries its count and a 95% Wilson interval on the observed rate,
    and ``interpretable`` is False where the count is too small to reason from.
    Ties in the score can make bin sizes differ by a few patients; that is
    preferred over splitting identical predictions across bins.
    """
    y_true_arr, y_prob_arr = _clean(y_true, y_prob)
    if y_true_arr.size == 0:
        return []

    order = np.argsort(y_prob_arr, kind="mergesort")
    # np.array_split gives sizes differing by at most one.
    rows: list[dict[str, Any]] = []
    for b, chunk in enumerate(np.array_split(order, n_bins)):
        if chunk.size == 0:
            continue
        truths = y_true_arr[chunk]
        probs = y_prob_arr[chunk]
        successes = int(truths.sum())
        lower, upper = wilson_interval(successes, int(chunk.size))
        rows.append(
            {
                "bin": b,
                "n": int(chunk.size),
                "n_positive": successes,
                "predicted_lower": float(probs.min()),
                "predicted_upper": float(probs.max()),
                "mean_predicted": float(probs.mean()),
                "observed_rate": float(truths.mean()),
                "observed_ci_lower": float(lower),
                "observed_ci_upper": float(upper),
                "gap": float(truths.mean() - probs.mean()),
                "interpretable": bool(chunk.size >= MIN_INTERPRETABLE_BIN_COUNT),
            }
        )
    return rows


class PlattCalibrator:
    """Logistic recalibration of a score, fitted once and then frozen.

    Platt scaling rather than isotonic: isotonic needs several hundred samples per
    step to avoid fitting noise into a staircase, and with 432 training patients it
    would produce a calibration map that looks precise and is not. Platt has two
    parameters and degrades to "no correction" gracefully.

    The fit is guarded by ``source``. A calibrator fitted on the predictions it is
    later evaluated against reports its own training error as generalization, and
    that mistake is invisible in the resulting numbers -- so it is refused here
    rather than left to reviewer discipline.
    """

    def __init__(self) -> None:
        self.coef_: float | None = None
        self.intercept_: float | None = None
        self.n_fit_: int = 0
        self.fit_source_: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    def fit(
        self,
        y_true: Sequence[float] | np.ndarray,
        y_prob: Sequence[float] | np.ndarray,
        *,
        source: str,
    ) -> PlattCalibrator:
        """Fit on out-of-fold training predictions.

        Args:
            y_true: Labels of the *training* patients.
            y_prob: Out-of-fold predictions for those same patients.
            source: Must be ``ALLOWED_CALIBRATION_FIT_SOURCE``. Anything else --
                in particular held-out predictions -- is refused.

        Raises:
            ValueError: If ``source`` is not the allowed one, or if the data
                cannot support a fit.
        """
        if source != ALLOWED_CALIBRATION_FIT_SOURCE:
            raise ValueError(
                f"PlattCalibrator may only be fitted on '{ALLOWED_CALIBRATION_FIT_SOURCE}' "
                f"predictions, got '{source}'. Fitting on held-out predictions would make "
                "the held-out Brier score a training-set number wearing a held-out label."
            )
        y_true_arr, y_prob_arr = _clean(y_true, y_prob)
        if y_true_arr.size < 20 or len(np.unique(y_true_arr)) < 2:
            raise ValueError(
                "Refusing to fit a calibrator on fewer than 20 out-of-fold predictions "
                "or on a single-class outcome; the resulting map would be noise."
            )
        logits = _logit(y_prob_arr).reshape(-1, 1)
        model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        model.fit(logits, y_true_arr.astype(int))
        self.coef_ = float(model.coef_[0][0])
        self.intercept_ = float(model.intercept_[0])
        self.n_fit_ = int(y_true_arr.size)
        self.fit_source_ = source
        return self

    def transform(self, y_prob: Sequence[float] | np.ndarray) -> np.ndarray:
        """Apply the frozen calibrator. Never re-fits, whatever it is handed."""
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator.fit() must be called before transform().")
        logits = _logit(np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0))
        assert self.coef_ is not None and self.intercept_ is not None
        return 1.0 / (1.0 + np.exp(-(self.coef_ * logits + self.intercept_)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "platt_scaling",
            "coef": self.coef_,
            "intercept": self.intercept_,
            "n_fit": self.n_fit_,
            "fit_source": self.fit_source_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlattCalibrator:
        """Rehydrate a frozen calibrator written by :meth:`to_dict`.

        This is a load path, not a fit path: it restores two already-fitted
        parameters and deliberately bypasses :meth:`fit`, whose ``source`` guard
        exists to police what a calibrator may be *trained* on. Re-running that
        guard here would be meaningless -- the training data is long gone -- so
        the provenance it recorded is carried across instead, and a payload that
        never passed the guard in the first place is refused.
        """
        method = payload.get("method")
        if method != "platt_scaling":
            raise ValueError(f"Expected a platt_scaling calibrator, got {method!r}.")
        coef, intercept = payload.get("coef"), payload.get("intercept")
        if coef is None or intercept is None:
            raise ValueError("Calibrator payload is missing 'coef' or 'intercept'.")
        source = payload.get("fit_source")
        if source != ALLOWED_CALIBRATION_FIT_SOURCE:
            raise ValueError(
                f"Refusing to load a calibrator recorded as fitted on {source!r}; only "
                f"{ALLOWED_CALIBRATION_FIT_SOURCE!r} is admissible. A calibrator fitted "
                "on held-out predictions would report its own training error as "
                "generalization, and loading it would launder that mistake into serving."
            )
        calibrator = cls()
        calibrator.coef_ = float(coef)
        calibrator.intercept_ = float(intercept)
        calibrator.n_fit_ = int(payload.get("n_fit", 0))
        calibrator.fit_source_ = source
        return calibrator


def discrimination_report(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """AUROC, AUPRC and Brier -- the three retained headline numbers.

    AUROC and AUPRC are invariant to any monotone recalibration, so they are
    reported once and are identical for raw and calibrated scores. Brier is not,
    which is exactly why it is the score to compare across the two.
    """
    y_true_arr, y_prob_arr = _clean(y_true, y_prob)
    if y_true_arr.size == 0 or len(np.unique(y_true_arr)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    labels = y_true_arr.astype(int)
    return {
        "auroc": float(roc_auc_score(labels, y_prob_arr)),
        "auprc": float(average_precision_score(labels, y_prob_arr)),
        "brier": float(np.mean((y_prob_arr - y_true_arr) ** 2)),
        "n": float(y_true_arr.size),
        "positive_rate": float(labels.mean()),
    }


def simplified_calibration_report(
    y_true: Sequence[float] | np.ndarray,
    raw_prob: Sequence[float] | np.ndarray,
    *,
    calibrator: PlattCalibrator | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """The calibration block written into the report.

    Keeps both scores when a calibrator is supplied: the raw score is what the
    model emits and what every earlier result was computed from, and dropping it
    would make the two incomparable.
    """
    report: dict[str, Any] = {
        "raw": {
            **discrimination_report(y_true, raw_prob),
            "reliability_bins": equal_frequency_bins(y_true, raw_prob, n_bins=n_bins),
        },
        "n_bins": n_bins,
        "binning": "equal_frequency",
        "min_interpretable_bin_count": MIN_INTERPRETABLE_BIN_COUNT,
        "calibrated": None,
        "calibrator": None,
        "caveats": [
            "Bins with `interpretable: false` carry too few patients to support a "
            "claim about calibration in that score range; read the confidence "
            "interval, not the point estimate.",
            "Calibration is uncertain wherever the cohort is sparsely represented, "
            "which on this cohort is the middle of the score range.",
        ],
    }
    if calibrator is not None and calibrator.is_fitted:
        calibrated = calibrator.transform(raw_prob)
        report["calibrated"] = {
            **discrimination_report(y_true, calibrated),
            "reliability_bins": equal_frequency_bins(y_true, calibrated, n_bins=n_bins),
        }
        report["calibrator"] = calibrator.to_dict()
    else:
        report["caveats"].append(
            "No calibrator was applied: only the raw model score is reported."
        )
    return report


__all__ = [
    "ALLOWED_CALIBRATION_FIT_SOURCE",
    "DEFAULT_N_BINS",
    "MIN_INTERPRETABLE_BIN_COUNT",
    "PlattCalibrator",
    "brier_score",
    "calibration_curve_points",
    "calibration_metrics_dict",
    "calibration_report",
    "calibration_slope_intercept",
    "discrimination_report",
    "equal_frequency_bins",
    "expected_calibration_error",
    "simplified_calibration_report",
    "wilson_interval",
]
