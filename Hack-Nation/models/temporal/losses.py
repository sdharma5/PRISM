"""The multi-task state loss.

    L_state = lambda_h * L_hormone + lambda_c * L_cycle
            + lambda_s * L_symptom + lambda_m * L_masked

The four terms are not interchangeable, and the reason to train them jointly is
that each constrains a different failure of the others:

* **L_hormone** anchors the embedding to measurable physiology, so the state is
  not a free-floating latent nobody can check.
* **L_cycle** supplies the coarse temporal structure that hormones alone leave
  ambiguous on sparse days.
* **L_symptom** forces the representation to carry information about how a person
  actually feels, which is the outcome that matters to them.
* **L_masked** is self-supervised: values are hidden and reconstructed, which
  teaches the model to interpolate through the heavy, non-random missingness
  instead of relying on days that happen to be observed.

Every loss here is computed **over observed entries only**. Scoring an unobserved
value against zero would train the model to predict the testing schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-9


@dataclass
class LossWeights:
    """Weights of the four state-loss terms."""

    hormone: float = 1.0
    cycle: float = 1.0
    symptom: float = 0.5
    masked: float = 0.5


def _observed_mask(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    """Default to 'everything observed' when no mask is supplied."""
    if mask is None:
        return np.ones(shape, dtype=float)
    return np.asarray(mask, dtype=float)


def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared error over observed entries only."""
    p, t = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    m = _observed_mask(mask, t.shape)
    denom = float(m.sum())
    if denom < EPS:
        return 0.0
    return float((m * (p - t) ** 2).sum() / denom)


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean absolute error over observed entries only."""
    p, t = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    m = _observed_mask(mask, t.shape)
    denom = float(m.sum())
    if denom < EPS:
        return 0.0
    return float((m * np.abs(p - t)).sum() / denom)


def gaussian_nll(
    pred_mean: np.ndarray,
    pred_log_var: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Gaussian negative log-likelihood with a predicted variance.

    Preferred over MSE when the model must express *how unsure* it is. Hormone
    values on a day with no observation for a week deserve a wide interval, and
    only a likelihood loss gives the model a reason to widen it.
    """
    mu = np.asarray(pred_mean, dtype=float)
    log_var = np.clip(np.asarray(pred_log_var, dtype=float), -10.0, 10.0)
    t = np.asarray(target, dtype=float)
    m = _observed_mask(mask, t.shape)
    denom = float(m.sum())
    if denom < EPS:
        return 0.0
    per = 0.5 * (log_var + (t - mu) ** 2 / np.exp(log_var) + np.log(2.0 * np.pi))
    return float((m * per).sum() / denom)


def hormone_loss(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    kind: str = "mse",
    pred_log_var: np.ndarray | None = None,
) -> float:
    """L_hormone over LH / E3G / PdG, observed entries only.

    Args:
        pred: Predicted values ``(N, C)``.
        target: True values ``(N, C)``.
        mask: 1 where the target is observed.
        kind: ``"mse"``, ``"mae"`` or ``"gaussian_nll"``.
        pred_log_var: Required for ``"gaussian_nll"``.

    Returns:
        Scalar loss.
    """
    if kind == "mse":
        return masked_mse(pred, target, mask)
    if kind == "mae":
        return masked_mae(pred, target, mask)
    if kind == "gaussian_nll":
        if pred_log_var is None:
            raise ValueError("gaussian_nll requires pred_log_var.")
        return gaussian_nll(pred, pred_log_var, target, mask)
    raise ValueError(f"Unknown hormone loss kind '{kind}'.")


def categorical_cross_entropy(
    probs: np.ndarray, targets: np.ndarray, *, class_weights: np.ndarray | None = None
) -> float:
    """L_cycle: cross-entropy over the four cycle-phase classes.

    Class weights matter here because the peri-ovulatory phase is a ~5-day window
    in a ~28-day cycle. Unweighted training reaches a good accuracy by never
    predicting it, which destroys the one phase call people most want.

    Args:
        probs: Predicted probabilities ``(N, K)``.
        targets: Integer class indices ``(N,)``.
        class_weights: Optional per-class weights ``(K,)``.

    Returns:
        Scalar loss.
    """
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    y = np.asarray(targets, dtype=int)
    if y.size == 0:
        return 0.0
    picked = p[np.arange(y.size), y]
    per = -np.log(picked)
    if class_weights is not None:
        per = per * np.asarray(class_weights, dtype=float)[y]
    return float(per.mean())


def binary_cross_entropy(
    probs: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """L_symptom: multilabel BCE over observed symptom reports."""
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(targets, dtype=float)
    m = _observed_mask(mask, y.shape)
    denom = float(m.sum())
    if denom < EPS:
        return 0.0
    per = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float((m * per).sum() / denom)


def masked_reconstruction_loss(
    pred: np.ndarray, target: np.ndarray, artificial_mask: np.ndarray
) -> float:
    """L_masked: reconstruct values that were artificially hidden.

    Scored *only* on the artificially masked entries, since reconstructing the
    values the model was shown is trivial and carries no learning signal.
    """
    return masked_mse(pred, target, artificial_mask)


def make_artificial_mask(
    observed_mask: np.ndarray, *, mask_fraction: float = 0.2, seed: int = 0
) -> np.ndarray:
    """Randomly hide a fraction of the OBSERVED entries for self-supervision.

    Only observed entries can be hidden, because an already-missing entry has no
    ground truth to reconstruct.
    """
    rng = np.random.default_rng(seed)
    observed = np.asarray(observed_mask, dtype=bool)
    draw = rng.random(observed.shape) < mask_fraction
    return (observed & draw).astype(float)


def state_loss(
    *,
    hormone_pred: np.ndarray | None = None,
    hormone_target: np.ndarray | None = None,
    hormone_mask: np.ndarray | None = None,
    hormone_log_var: np.ndarray | None = None,
    hormone_kind: str = "mse",
    cycle_probs: np.ndarray | None = None,
    cycle_target: np.ndarray | None = None,
    cycle_class_weights: np.ndarray | None = None,
    symptom_probs: np.ndarray | None = None,
    symptom_target: np.ndarray | None = None,
    symptom_mask: np.ndarray | None = None,
    masked_pred: np.ndarray | None = None,
    masked_target: np.ndarray | None = None,
    masked_mask: np.ndarray | None = None,
    weights: LossWeights | None = None,
) -> dict[str, float]:
    """Compute L_state and every component.

    Components are returned individually so training logs show which head is
    driving the total, rather than one number that hides a collapsed head.

    Returns:
        ``{"hormone", "cycle", "symptom", "masked", "total"}``.
    """
    w = weights or LossWeights()
    components = {"hormone": 0.0, "cycle": 0.0, "symptom": 0.0, "masked": 0.0}

    if hormone_pred is not None and hormone_target is not None:
        components["hormone"] = hormone_loss(
            hormone_pred,
            hormone_target,
            hormone_mask,
            kind=hormone_kind,
            pred_log_var=hormone_log_var,
        )
    if cycle_probs is not None and cycle_target is not None:
        components["cycle"] = categorical_cross_entropy(
            cycle_probs, cycle_target, class_weights=cycle_class_weights
        )
    if symptom_probs is not None and symptom_target is not None:
        components["symptom"] = binary_cross_entropy(symptom_probs, symptom_target, symptom_mask)
    if masked_pred is not None and masked_target is not None and masked_mask is not None:
        components["masked"] = masked_reconstruction_loss(masked_pred, masked_target, masked_mask)

    total = (
        w.hormone * components["hormone"]
        + w.cycle * components["cycle"]
        + w.symptom * components["symptom"]
        + w.masked * components["masked"]
    )
    return {**components, "total": float(total)}


# --------------------------------------------------------------------------
# torch variants
# --------------------------------------------------------------------------


def _torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Torch losses require the optional 'torch' extra.") from exc
    return torch


def masked_mse_torch(pred: Any, target: Any, mask: Any | None = None) -> Any:
    """Torch twin of :func:`masked_mse`."""
    torch = _torch()
    if mask is None:
        mask = torch.ones_like(target)
    denom = mask.sum().clamp_min(1.0)
    return (mask * (pred - target) ** 2).sum() / denom


def gaussian_nll_torch(mean: Any, log_var: Any, target: Any, mask: Any | None = None) -> Any:
    """Torch twin of :func:`gaussian_nll`."""
    torch = _torch()
    if mask is None:
        mask = torch.ones_like(target)
    log_var = log_var.clamp(-10.0, 10.0)
    per = 0.5 * (log_var + (target - mean) ** 2 / log_var.exp() + float(np.log(2.0 * np.pi)))
    return (mask * per).sum() / mask.sum().clamp_min(1.0)


def state_loss_torch(
    *,
    hormone_pred: Any = None,
    hormone_target: Any = None,
    hormone_mask: Any = None,
    cycle_logits: Any = None,
    cycle_target: Any = None,
    symptom_logits: Any = None,
    symptom_target: Any = None,
    symptom_mask: Any = None,
    masked_pred: Any = None,
    masked_target: Any = None,
    masked_mask: Any = None,
    weights: LossWeights | None = None,
) -> Any:
    """Torch twin of :func:`state_loss`, returning the scalar total."""
    torch = _torch()
    w = weights or LossWeights()
    total = torch.zeros((), dtype=torch.float32)
    if hormone_pred is not None and hormone_target is not None:
        total = total + w.hormone * masked_mse_torch(hormone_pred, hormone_target, hormone_mask)
    if cycle_logits is not None and cycle_target is not None:
        total = total + w.cycle * torch.nn.functional.cross_entropy(
            cycle_logits, cycle_target.long()
        )
    if symptom_logits is not None and symptom_target is not None:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            symptom_logits, symptom_target.float(), reduction="none"
        )
        if symptom_mask is not None:
            bce = (bce * symptom_mask).sum() / symptom_mask.sum().clamp_min(1.0)
        else:
            bce = bce.mean()
        total = total + w.symptom * bce
    if masked_pred is not None and masked_target is not None:
        total = total + w.masked * masked_mse_torch(masked_pred, masked_target, masked_mask)
    return total
