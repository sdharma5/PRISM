"""L_outside must be > 0 exactly when follicle probability sits outside the ovary.

    L_outside = sum_i P_i(follicle) * (1 - P_i(ovary))

A follicle is by definition intra-ovarian, so any follicle probability mass at a
voxel the model also believes is extra-ovarian is an anatomically impossible
claim. This penalty needs no ground truth, which is why it can regularise
unlabelled studies — and why its exact behaviour is worth pinning down.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.ultrasound.losses import (
    CLASS_BACKGROUND,
    CLASS_FOLLICLE,
    CLASS_OVARY,
    boundary_loss_numpy,
    cross_entropy_numpy,
    dice_loss_numpy,
    outside_penalty_from_probs_numpy,
    outside_penalty_numpy,
    quality_head_loss_numpy,
    softmax_numpy,
    total_segmentation_loss_numpy,
)


def test_zero_when_all_follicle_mass_is_inside_the_ovary():
    """Follicle probability inside the ovary incurs no penalty."""
    follicle = np.array([0.0, 0.9, 0.7, 0.0])
    ovary = np.array([0.0, 1.0, 1.0, 0.0])
    assert outside_penalty_numpy(follicle, ovary) == pytest.approx(0.0)


def test_positive_when_follicle_mass_is_outside_the_ovary():
    """Follicle probability where the model says 'not ovary' is penalised."""
    follicle = np.array([0.0, 0.0, 0.8, 0.0])
    ovary = np.array([1.0, 1.0, 0.0, 1.0])
    penalty = outside_penalty_numpy(follicle, ovary)
    assert penalty > 0.0
    assert penalty == pytest.approx(0.8)


def test_matches_the_specified_formula_exactly():
    """The implementation is literally sum_i P_f * (1 - P_o)."""
    rng = np.random.default_rng(0)
    follicle = rng.random(50)
    ovary = rng.random(50)
    expected = float(np.sum(follicle * (1.0 - ovary)))
    assert outside_penalty_numpy(follicle, ovary) == pytest.approx(expected)


def test_penalty_scales_with_the_amount_of_misplaced_mass():
    """More misplaced follicle probability means a larger penalty."""
    ovary = np.zeros(10)
    small = outside_penalty_numpy(np.full(10, 0.1), ovary)
    large = outside_penalty_numpy(np.full(10, 0.9), ovary)
    assert large > small


def test_zero_when_no_follicle_is_predicted_anywhere():
    """No follicle prediction means nothing can be misplaced."""
    assert outside_penalty_numpy(np.zeros(20), np.zeros(20)) == pytest.approx(0.0)


def test_three_class_wrapper_treats_follicle_voxels_as_ovarian():
    """In a mutually exclusive 3-class head, a follicle voxel is also ovary.

    Without this, a confident follicle prediction would penalise *itself*: the
    softmax pushes P(ovary) down exactly where P(follicle) is high.
    """
    probs = np.zeros((3, 4))
    probs[CLASS_FOLLICLE, :] = 1.0  # certain follicle, so P(ovary stroma) = 0
    assert outside_penalty_from_probs_numpy(probs) == pytest.approx(0.0)


def test_three_class_wrapper_penalises_follicle_in_background():
    """Follicle probability where background dominates is penalised."""
    probs = np.zeros((3, 4))
    probs[CLASS_BACKGROUND, :] = 0.6
    probs[CLASS_FOLLICLE, :] = 0.4
    penalty = outside_penalty_from_probs_numpy(probs)
    assert penalty == pytest.approx(4 * 0.4 * 0.6)


def test_reduction_modes_agree_up_to_the_element_count():
    """``mean`` is ``sum`` divided by the number of voxels."""
    rng = np.random.default_rng(1)
    follicle, ovary = rng.random(30), rng.random(30)
    total = outside_penalty_numpy(follicle, ovary, reduction="sum")
    mean = outside_penalty_numpy(follicle, ovary, reduction="mean")
    assert mean == pytest.approx(total / 30)


def test_shape_mismatch_is_rejected():
    """Silently broadcasting mismatched shapes would hide a real bug."""
    with pytest.raises(ValueError, match="shape mismatch"):
        outside_penalty_numpy(np.zeros(4), np.zeros(5))


def test_dice_and_cross_entropy_reward_the_correct_segmentation():
    """A perfect prediction must score better than an inverted one."""
    target = np.zeros((6, 6, 6), dtype=int)
    target[1:5, 1:5, 1:5] = CLASS_OVARY
    target[2:4, 2:4, 2:4] = CLASS_FOLLICLE

    perfect = np.zeros((3, 6, 6, 6))
    for c in range(3):
        perfect[c] = (target == c).astype(float)
    perfect = np.clip(perfect, 1e-4, 1.0)
    perfect /= perfect.sum(axis=0, keepdims=True)

    wrong = np.full((3, 6, 6, 6), 1.0 / 3.0)

    assert dice_loss_numpy(perfect, target) < dice_loss_numpy(wrong, target)
    assert cross_entropy_numpy(perfect, target) < cross_entropy_numpy(wrong, target)


def test_boundary_loss_is_lower_for_a_correct_ovary_boundary():
    """Distance weighting must penalise probability mass far from the truth."""
    target = np.zeros((8, 8, 8), dtype=int)
    target[2:6, 2:6, 2:6] = CLASS_OVARY

    good = np.zeros((3, 8, 8, 8))
    good[CLASS_OVARY] = (target == CLASS_OVARY).astype(float)
    bad = np.zeros((3, 8, 8, 8))
    bad[CLASS_OVARY] = 1.0 - (target == CLASS_OVARY).astype(float)

    assert boundary_loss_numpy(good, target) < boundary_loss_numpy(bad, target)


def test_total_loss_reports_every_component():
    """Training logs need components, not one opaque number."""
    target = np.zeros((6, 6, 6), dtype=int)
    target[1:5, 1:5, 1:5] = CLASS_OVARY
    probs = softmax_numpy(np.random.default_rng(0).random((3, 6, 6, 6)), axis=0)
    result = total_segmentation_loss_numpy(probs, target)
    assert set(result) == {"dice", "cross_entropy", "boundary", "outside", "total"}
    assert all(np.isfinite(v) for v in result.values())


def test_quality_head_loss_is_minimised_by_the_right_answer():
    """The quality head loss must actually prefer the correct flags."""
    target = {"ovary_visible": 1.0, "pixel_spacing_available": 0.0, "overall_quality_score": 0.8}
    good = {"ovary_visible": 0.95, "pixel_spacing_available": 0.05, "overall_quality_score": 0.8}
    bad = {"ovary_visible": 0.05, "pixel_spacing_available": 0.95, "overall_quality_score": 0.1}
    assert quality_head_loss_numpy(good, target) < quality_head_loss_numpy(bad, target)
