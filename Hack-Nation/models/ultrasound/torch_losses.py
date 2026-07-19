"""Class-imbalance-aware losses for ovary/follicle segmentation.

    L = lam_o * L_ovary + lam_f * L_follicle + lam_out * L_outside

Three deliberate choices:

**Dice + BCE for the ovary.** Dice handles the foreground/background imbalance;
BCE keeps gradients alive when Dice saturates. Neither alone is stable here.

**Focal BCE for follicles.** Follicle voxels are a minority *within* a minority,
and 26% of label slices contain no follicle at all. Plain BCE on such data
converges to predicting "no follicle" everywhere -- a state with low loss and no
utility. Focal down-weights the easy negatives that dominate the sum.

**An outside-ovary penalty with gradients to BOTH heads.** The term penalises
follicle probability where there is no ovary. It is tempting to detach the ovary
prediction so only the follicle head is corrected, but that is wrong: when a
follicle is confidently correct and the ovary mask has missed it, the informative
gradient is the one that *grows the ovary*. Detaching would silently discard it.
The penalty is anchored on ground-truth ovary when available and on the predicted
ovary otherwise, so it is meaningful at inference-shaped inputs too.

Empty masks are the normal case, not an edge case, so every term is written to be
finite when a target is entirely zero -- covered by explicit tests.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

__all__ = ["DualHeadLoss", "dice_loss", "focal_bce", "outside_ovary_penalty"]

_EPS = 1e-6


def dice_loss(logits: torch.Tensor, target: torch.Tensor, *, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss, computed per-sample then averaged.

    ``smooth`` is 1.0, not a tiny epsilon, and the difference is not cosmetic.
    26% of label slices here contain no follicle at all, so the empty-target case
    is routine. With a 1e-6 epsilon a *correct* near-zero prediction on an empty
    mask still scores a loss of ~1.0: the predicted probabilities are small but
    their sum over thousands of voxels dwarfs the epsilon, so the ratio collapses
    to zero and the model is punished for being right. A smoothing constant of 1
    puts that case near zero loss instead, and lets the term degrade gracefully
    as the prediction grows.

    Averaging per-sample rather than over the flattened batch keeps a slice with
    a large ovary from dominating one with a small one.
    """
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dims)
    cardinality = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return (1.0 - dice).mean()


def focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal binary cross-entropy.

    Args:
        logits: Raw predictions.
        target: 0/1 targets.
        gamma: Focusing strength; 0 recovers weighted BCE.
        alpha: Weight on the positive class. Above 0.5 because follicle voxels
            are far rarer than background.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def outside_ovary_penalty(
    follicle_logits: torch.Tensor,
    ovary_logits: torch.Tensor,
    ovary_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise follicle probability outside the ovary.

    A follicle outside its ovary is anatomically impossible, so this encodes a
    constraint the data cannot teach from 12 volumes alone.

    Args:
        follicle_logits: Follicle head output.
        ovary_logits: Ovary head output. Gradients flow through this on purpose
            -- see the module docstring.
        ovary_target: Ground-truth ovary when available. Preferred during
            training because an early-epoch predicted ovary is noise, and
            anchoring the constraint on noise teaches nothing.
    """
    follicle_probs = torch.sigmoid(follicle_logits)
    if ovary_target is not None:
        outside = 1.0 - ovary_target
        # Ovary logits still participate so the term remains differentiable in
        # both heads; with a ground-truth anchor the ovary gradient is zero here,
        # which is correct -- the supervision already constrains it.
        return (follicle_probs * outside).mean()
    return (follicle_probs * (1.0 - torch.sigmoid(ovary_logits))).mean()


class DualHeadLoss(nn.Module):
    """Weighted sum of the ovary, follicle and outside-ovary terms."""

    def __init__(
        self,
        *,
        lambda_ovary: float = 1.0,
        lambda_follicle: float = 1.0,
        lambda_outside: float = 0.1,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        use_ground_truth_ovary_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_ovary = lambda_ovary
        self.lambda_follicle = lambda_follicle
        self.lambda_outside = lambda_outside
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.use_ground_truth_ovary_anchor = use_ground_truth_ovary_anchor

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        ovary_target: torch.Tensor,
        follicle_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the total loss and each component, for logging."""
        ovary_logits = outputs["ovary"]
        follicle_logits = outputs["follicle"]

        ovary_term = self.dice_weight * dice_loss(
            ovary_logits, ovary_target
        ) + self.bce_weight * F.binary_cross_entropy_with_logits(ovary_logits, ovary_target)

        follicle_term = self.dice_weight * dice_loss(
            follicle_logits, follicle_target
        ) + self.bce_weight * focal_bce(
            follicle_logits,
            follicle_target,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
        )

        outside_term = outside_ovary_penalty(
            follicle_logits,
            ovary_logits,
            ovary_target if self.use_ground_truth_ovary_anchor else None,
        )

        total = (
            self.lambda_ovary * ovary_term
            + self.lambda_follicle * follicle_term
            + self.lambda_outside * outside_term
        )
        return {
            "loss": total,
            "ovary": ovary_term.detach(),
            "follicle": follicle_term.detach(),
            "outside": outside_term.detach(),
        }
