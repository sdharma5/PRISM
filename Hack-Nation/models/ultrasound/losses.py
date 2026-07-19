"""Losses for ovarian ultrasound segmentation, with numpy and torch variants.

The numpy variants are the reference implementations: they are unit-tested and
readable, and they let the loss semantics be verified in an environment without
torch. The torch variants must stay numerically equivalent.

The scientifically important term here is :func:`outside_penalty_numpy`. A
follicle is *by definition* an intra-ovarian structure. A network that predicts
follicle probability at a voxel it also believes is outside the ovary is making
an anatomically impossible claim, and such voxels are the dominant source of
false antral-follicle counts. The penalty is

    L_outside = sum_i P_i(follicle) * (1 - P_i(ovary))

which is exactly zero when every follicle-probability mass sits inside the
predicted ovary, and grows with the amount of misplaced mass. It needs no ground
truth, so it regularises even unlabelled studies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-6

#: Semantic class indices shared by every ultrasound segmentation component.
CLASS_BACKGROUND = 0
CLASS_OVARY = 1
CLASS_FOLLICLE = 2
N_CLASSES = 3


# --------------------------------------------------------------------------
# numpy reference implementations
# --------------------------------------------------------------------------


def softmax_numpy(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax over ``axis``."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), EPS, None)


def one_hot_numpy(target: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """One-hot encode an integer label volume to ``(C, ...)``."""
    target = np.asarray(target, dtype=int)
    out = np.zeros((n_classes, *target.shape), dtype=float)
    for c in range(n_classes):
        out[c] = target == c
    return out


def dice_loss_numpy(
    probs: np.ndarray,
    target: np.ndarray,
    *,
    classes: tuple[int, ...] = (CLASS_OVARY, CLASS_FOLLICLE),
    smooth: float = 1.0,
) -> float:
    """Soft Dice loss averaged over the requested foreground classes.

    Dice is used alongside cross-entropy because follicles occupy a tiny voxel
    fraction; a pure cross-entropy optimum is the empty prediction.

    Args:
        probs: Class probabilities, shape ``(C, ...)``.
        target: Integer label volume.
        classes: Foreground class indices to average over.
        smooth: Laplace smoothing, which keeps the loss finite for empty classes.

    Returns:
        Scalar loss in [0, 1].
    """
    onehot = one_hot_numpy(target, probs.shape[0])
    losses = []
    for c in classes:
        p, g = probs[c].ravel(), onehot[c].ravel()
        intersection = float((p * g).sum())
        denom = float(p.sum() + g.sum())
        losses.append(1.0 - (2.0 * intersection + smooth) / (denom + smooth))
    return float(np.mean(losses)) if losses else 0.0


def cross_entropy_numpy(
    probs: np.ndarray,
    target: np.ndarray,
    *,
    class_weights: tuple[float, ...] | None = None,
) -> float:
    """Voxelwise weighted cross-entropy on already-softmaxed probabilities."""
    onehot = one_hot_numpy(target, probs.shape[0])
    weights = np.asarray(
        class_weights if class_weights is not None else [1.0] * probs.shape[0], dtype=float
    ).reshape((-1,) + (1,) * (probs.ndim - 1))
    log_p = np.log(np.clip(probs, EPS, 1.0))
    per_voxel = -(weights * onehot * log_p).sum(axis=0)
    return float(per_voxel.mean())


def boundary_loss_numpy(
    probs: np.ndarray,
    target: np.ndarray,
    *,
    class_index: int = CLASS_OVARY,
) -> float:
    """Distance-weighted boundary loss for one class.

    Ovarian volume error is dominated by boundary placement, not by interior
    voxels, so a term whose gradient is largest far from the true boundary
    penalises exactly the errors that move the reported volume.

    Args:
        probs: Class probabilities ``(C, ...)``.
        target: Integer label volume.
        class_index: Class whose boundary is supervised.

    Returns:
        Mean signed-distance-weighted probability mass, >= 0 in practice.
    """
    mask = np.asarray(target) == class_index
    distance = _signed_distance(mask)
    return float((probs[class_index] * distance).mean())


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance map: negative inside the mask, positive outside."""
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: PLC0415
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return (~mask).astype(float)
    if not mask.any():
        return np.ones(mask.shape, dtype=float)
    if mask.all():
        return -np.ones(mask.shape, dtype=float)
    outside = distance_transform_edt(~mask)
    inside = distance_transform_edt(mask)
    return outside - inside


def outside_penalty_numpy(
    follicle_prob: np.ndarray,
    ovary_prob: np.ndarray,
    *,
    reduction: str = "sum",
) -> float:
    """L_outside = sum_i P_i(follicle) * (1 - P_i(ovary)).

    Zero exactly when no follicle probability mass lies outside the predicted
    ovary. Requires no labels, so it constrains the anatomically impossible
    configuration on every study including unlabelled ones.

    Args:
        follicle_prob: Per-voxel follicle probability.
        ovary_prob: Per-voxel ovary probability (the ovary *region*, which for a
            mutually exclusive 3-class head should include follicle voxels; pass
            ``P(ovary) + P(follicle)`` in that case).
        reduction: ``"sum"`` (as specified) or ``"mean"``.

    Returns:
        Non-negative scalar penalty.
    """
    f = np.asarray(follicle_prob, dtype=float)
    o = np.asarray(ovary_prob, dtype=float)
    if f.shape != o.shape:
        raise ValueError(f"shape mismatch: follicle {f.shape} vs ovary {o.shape}")
    per_voxel = f * (1.0 - o)
    if reduction == "mean":
        return float(per_voxel.mean())
    if reduction == "sum":
        return float(per_voxel.sum())
    raise ValueError(f"Unknown reduction '{reduction}'.")


def outside_penalty_from_probs_numpy(probs: np.ndarray, *, reduction: str = "sum") -> float:
    """Convenience wrapper for a ``(3, ...)`` softmax output.

    "Inside the ovary" is ``P(ovary) + P(follicle)``, because in a mutually
    exclusive 3-class parameterisation a voxel labelled follicle is also, in
    reality, ovarian tissue.
    """
    return outside_penalty_numpy(
        probs[CLASS_FOLLICLE],
        probs[CLASS_OVARY] + probs[CLASS_FOLLICLE],
        reduction=reduction,
    )


def quality_head_loss_numpy(
    predicted: dict[str, float],
    target: dict[str, float],
    *,
    binary_keys: tuple[str, ...] = (
        "ovary_visible",
        "whole_ovary_visible",
        "laterality_available",
        "pixel_spacing_available",
        "follicle_counting_feasible",
        "ovarian_volume_feasible",
    ),
    score_key: str = "overall_quality_score",
) -> float:
    """BCE over the binary quality flags plus MSE on the overall score.

    The quality head is what makes the pipeline able to abstain, so it is trained
    jointly rather than as a post-hoc filter.
    """
    terms: list[float] = []
    for key in binary_keys:
        if key not in target:
            continue
        p = float(np.clip(predicted.get(key, 0.5), EPS, 1 - EPS))
        y = float(target[key])
        terms.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    if score_key in target:
        terms.append(float((predicted.get(score_key, 0.0) - target[score_key]) ** 2))
    return float(np.mean(terms)) if terms else 0.0


def total_segmentation_loss_numpy(
    probs: np.ndarray,
    target: np.ndarray,
    *,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
    lambda_boundary: float = 0.1,
    lambda_outside: float = 0.01,
    class_weights: tuple[float, ...] | None = (0.2, 1.0, 3.0),
) -> dict[str, float]:
    """Weighted sum of all segmentation terms; returns every component.

    Components are returned individually so that training logs can show *which*
    constraint is binding, rather than a single opaque number.
    """
    dice = dice_loss_numpy(probs, target)
    ce = cross_entropy_numpy(probs, target, class_weights=class_weights)
    boundary = boundary_loss_numpy(probs, target)
    outside = outside_penalty_from_probs_numpy(probs, reduction="mean")
    total = (
        lambda_dice * dice + lambda_ce * ce + lambda_boundary * boundary + lambda_outside * outside
    )
    return {
        "dice": dice,
        "cross_entropy": ce,
        "boundary": boundary,
        "outside": outside,
        "total": float(total),
    }


# --------------------------------------------------------------------------
# torch variants (lazy import)
# --------------------------------------------------------------------------


def _torch() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Torch losses require the optional 'torch' extra: pip install '.[torch]'. "
            "The numpy variants in this module are always available."
        ) from exc
    return torch


def dice_loss_torch(
    logits: Any,
    target: Any,
    *,
    classes: tuple[int, ...] = (CLASS_OVARY, CLASS_FOLLICLE),
    smooth: float = 1.0,
) -> Any:
    """Soft Dice over ``(N, C, ...)`` logits and ``(N, ...)`` integer targets."""
    torch = _torch()
    probs = torch.softmax(logits, dim=1)
    losses = []
    for c in classes:
        p = probs[:, c].reshape(probs.shape[0], -1)
        g = (target == c).float().reshape(target.shape[0], -1)
        intersection = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        losses.append(1.0 - (2.0 * intersection + smooth) / (denom + smooth))
    return torch.stack(losses).mean()


def cross_entropy_torch(logits: Any, target: Any, *, class_weights: Any | None = None) -> Any:
    """Weighted voxelwise cross-entropy."""
    torch = _torch()
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, target.long(), weight=weight)


def boundary_loss_torch(logits: Any, distance_map: Any, *, class_index: int = CLASS_OVARY) -> Any:
    """Boundary loss given a precomputed signed distance map ``(N, ...)``."""
    torch = _torch()
    probs = torch.softmax(logits, dim=1)
    return (probs[:, class_index] * distance_map).mean()


def outside_penalty_torch(logits: Any, *, reduction: str = "sum") -> Any:
    """Torch twin of :func:`outside_penalty_from_probs_numpy`."""
    torch = _torch()
    probs = torch.softmax(logits, dim=1)
    inside = probs[:, CLASS_OVARY] + probs[:, CLASS_FOLLICLE]
    per_voxel = probs[:, CLASS_FOLLICLE] * (1.0 - inside)
    return per_voxel.sum() if reduction == "sum" else per_voxel.mean()


def quality_head_loss_torch(pred_logits: Any, pred_score: Any, targets: Any, score: Any) -> Any:
    """BCE-with-logits on the flags plus MSE on the overall score."""
    torch = _torch()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, targets.float())
    mse = torch.nn.functional.mse_loss(pred_score, score.float())
    return bce + mse


def total_segmentation_loss_torch(
    logits: Any,
    target: Any,
    *,
    distance_map: Any | None = None,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
    lambda_boundary: float = 0.1,
    lambda_outside: float = 0.01,
    class_weights: tuple[float, ...] | None = (0.2, 1.0, 3.0),
) -> Any:
    """Weighted sum matching :func:`total_segmentation_loss_numpy`."""
    total = lambda_dice * dice_loss_torch(logits, target) + lambda_ce * cross_entropy_torch(
        logits, target, class_weights=class_weights
    )
    if distance_map is not None:
        total = total + lambda_boundary * boundary_loss_torch(logits, distance_map)
    return total + lambda_outside * outside_penalty_torch(logits, reduction="mean")
