"""Calibrate soft profile-membership probabilities so they are not overconfident.

Scientific WHY
--------------
Both natural sources of a membership vector are systematically overconfident.
A softmax over distances to k-means centroids has an arbitrary scale — its
sharpness depends on the units of the representation, not on how sure we are.
GMM responsibilities are worse: they are conditional on the fitted mixture being
the true generative model, so a participant sitting between two components can
still be assigned 0.97 to one of them.

A reported "0.9 probability of resembling profile A" should mean that, in the
population of participants given 0.9, about 90% actually stay in profile A when
the analysis is repeated. We therefore calibrate against an *empirical
reproducibility target*: the per-participant bootstrap agreement rate from
:mod:`models.stability.bootstrap`. Temperature scaling is used because it is a
single-parameter, monotone transform — it cannot change who the dominant profile
is, only how confident the report is about it. That property matters: a
calibration step that silently reassigned patients would be indefensible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "CalibrationResult",
    "expected_calibration_error",
    "fit_temperature",
    "membership_from_distances",
    "temperature_scale",
]


def membership_from_distances(
    X: np.ndarray,
    centers: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Softmax over negative squared distance to each cluster centre.

    ``temperature`` > 1 flattens the distribution (less confident), < 1 sharpens
    it. The default of 1.0 is *uncalibrated* and should be replaced by
    :func:`fit_temperature` before anything is reported.
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling in log space to an existing probability matrix.

    Used for GMM responsibilities, where we already have probabilities rather
    than distances. Monotone in every coordinate, so the argmax is preserved.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    logits = np.log(np.clip(p, 1e-12, 1.0)) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def expected_calibration_error(
    confidences: Sequence[float] | np.ndarray,
    accuracies: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Binned |mean confidence − mean empirical agreement|, weighted by bin size.

    ``accuracies`` here are bootstrap *agreement rates*, not classification
    correctness: the "event" being calibrated is "this participant stays in this
    profile when the cohort is resampled".
    """
    conf = np.asarray(list(confidences), dtype=float)
    acc = np.asarray(list(accuracies), dtype=float)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(float(conf[mask].mean()) - float(acc[mask].mean()))
    return float(total)


@dataclass
class CalibrationResult:
    """A fitted temperature plus before/after calibration error."""

    temperature: float
    ece_before: float
    ece_after: float
    mean_confidence_before: float
    mean_confidence_after: float
    mean_agreement: float
    n_samples: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.ece_after <= self.ece_before


def fit_temperature(
    probabilities: np.ndarray,
    agreement_rates: Sequence[float],
    bounds: tuple[float, float] = (0.05, 50.0),
    n_bins: int = 10,
) -> CalibrationResult:
    """Fit the temperature that best matches confidence to bootstrap agreement.

    We minimize the expected calibration error between the top-1 membership
    probability and the participant's empirical bootstrap agreement rate. With a
    single free parameter on a bounded interval this is solved by scalar
    optimization; there is no risk of the calibration itself overfitting in any
    meaningful sense, but the result is still reported with before/after ECE so a
    reader can see whether it actually helped.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    agreement = np.asarray(list(agreement_rates), dtype=float)
    warnings: list[str] = []

    if p.shape[0] != agreement.shape[0]:
        raise ValueError("probabilities and agreement_rates must describe the same participants.")
    if p.shape[0] < 2:
        warnings.append("too few participants to fit a temperature; leaving it at 1.0")
        conf = p.max(axis=1) if p.size else np.array([1.0])
        ece = expected_calibration_error(conf, agreement, n_bins)
        return CalibrationResult(
            1.0,
            ece,
            ece,
            float(conf.mean()),
            float(conf.mean()),
            float(agreement.mean()) if agreement.size else 0.0,
            int(p.shape[0]),
            warnings,
        )

    def objective(log_t: float) -> float:
        scaled = temperature_scale(p, float(np.exp(log_t)))
        return expected_calibration_error(scaled.max(axis=1), agreement, n_bins)

    opt = minimize_scalar(
        objective,
        bounds=(float(np.log(bounds[0])), float(np.log(bounds[1]))),
        method="bounded",
    )
    temperature = float(np.exp(opt.x))

    conf_before = p.max(axis=1)
    conf_after = temperature_scale(p, temperature).max(axis=1)
    ece_before = expected_calibration_error(conf_before, agreement, n_bins)
    ece_after = expected_calibration_error(conf_after, agreement, n_bins)

    if ece_after > ece_before:
        warnings.append("temperature search did not improve ECE; reverting to T=1.0")
        temperature, conf_after, ece_after = 1.0, conf_before, ece_before
    if temperature < 1.0:
        warnings.append(
            f"fitted temperature {temperature:.3f} < 1 sharpens the distribution: the raw "
            "memberships were under-confident relative to bootstrap agreement."
        )

    return CalibrationResult(
        temperature=temperature,
        ece_before=ece_before,
        ece_after=ece_after,
        mean_confidence_before=float(conf_before.mean()),
        mean_confidence_after=float(conf_after.mean()),
        mean_agreement=float(agreement.mean()),
        n_samples=int(p.shape[0]),
        warnings=warnings,
    )
