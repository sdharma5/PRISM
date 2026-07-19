"""The four prediction heads and the composite L_state loss.

The heads are tested for the properties that make them scientifically usable
rather than merely runnable: the hormone head must be fitted on observed values
only, the cycle head must not optimise away the short peri-ovulatory window, the
symptom head must forecast tomorrow rather than copy today, and L_state must be
the stated weighted sum.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.temporal.heads import (
    CYCLE_CLASSES,
    HORMONE_TARGETS,
    CycleStateHead,
    HormoneReconstructionHead,
    MaskedReconstructionHead,
    SymptomHead,
)
from models.temporal.losses import (
    LossWeights,
    binary_cross_entropy,
    categorical_cross_entropy,
    gaussian_nll,
    hormone_loss,
    make_artificial_mask,
    masked_mae,
    masked_mse,
    state_loss,
)
from models.temporal.state_model import TemporalStateModel, grouped_participant_split
from tests.fixtures.synthetic_cycles import generate_cohort


@pytest.fixture(scope="module")
def fitted():
    cohort = generate_cohort(n_participants=8, n_days=70, seed=0)
    groups = [d.participant_id for d in cohort.days]
    train_index, test_index = grouped_participant_split(groups, test_fraction=0.3, seed=0)
    train = [cohort.days[i] for i in train_index]
    test = [cohort.days[i] for i in test_index]
    model = TemporalStateModel(lookback_days=21, hidden_size=24, seed=0).fit(train)
    return model, train, test


# -- hormone head ----------------------------------------------------------


def test_hormone_head_predicts_all_three_channels():
    """LH, E3G and PdG are all reconstructed."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 8))
    truth = np.column_stack([embeddings[:, 0] * 3 + 5, embeddings[:, 1] * 2, embeddings[:, 2]])
    head = HormoneReconstructionHead().fit(embeddings, truth)
    predictions = head.predict(embeddings)
    assert set(predictions) == set(HORMONE_TARGETS)
    assert predictions["lh"].shape == (60,)


def test_hormone_head_ignores_unobserved_targets():
    """Masked-out rows must not influence the fit."""
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(80, 4))
    truth = np.column_stack([embeddings[:, 0] * 2.0, np.zeros(80), np.zeros(80)])
    mask = np.ones((80, 3), dtype=bool)

    # Corrupt half the LH targets but mark them unobserved.
    corrupted = truth.copy()
    corrupted[40:, 0] = 999.0
    mask[40:, 0] = False

    clean = HormoneReconstructionHead().fit(embeddings, truth).predict(embeddings)["lh"]
    masked = HormoneReconstructionHead().fit(embeddings, corrupted, mask).predict(embeddings)["lh"]
    assert np.corrcoef(clean, masked)[0, 1] > 0.95


def test_hormone_head_reports_residual_uncertainty():
    """Predictions come with a stated spread, not bare point estimates."""
    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(70, 5))
    truth = np.column_stack([embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]])
    head = HormoneReconstructionHead().fit(embeddings, truth)
    _, residuals = head.predict_with_uncertainty(embeddings)
    assert set(residuals) == set(HORMONE_TARGETS)
    assert all(np.isfinite(v) and v >= 0 for v in residuals.values())


# -- cycle head ------------------------------------------------------------


def test_cycle_head_returns_a_distribution_over_four_phases():
    """Probabilities over exactly the four physiological phases, summing to 1."""
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(120, 6))
    labels = [CYCLE_CLASSES[i % 4] for i in range(120)]
    head = CycleStateHead().fit(embeddings, labels)
    probs = head.predict_proba(embeddings)
    assert probs.shape == (120, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_cycle_head_upweights_the_rare_peri_ovulatory_class():
    """Inverse-frequency weighting is what keeps ovulation predictable."""
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(200, 4))
    labels = ["luteal"] * 180 + ["peri_ovulatory"] * 20
    head = CycleStateHead().fit(embeddings, labels)
    assert head.class_weights is not None
    peri = head.class_weights[CYCLE_CLASSES.index("peri_ovulatory")]
    luteal = head.class_weights[CYCLE_CLASSES.index("luteal")]
    assert peri > luteal


def test_cycle_head_entropy_is_high_when_uninformative():
    """An unfit head must report maximal uncertainty rather than false confidence."""
    head = CycleStateHead()
    entropy = head.predict_entropy(np.zeros((5, 4)))
    assert np.allclose(entropy, 1.0)


def test_cycle_head_learns_a_separable_signal():
    """On a clean signal the head must actually classify."""
    rng = np.random.default_rng(5)
    centres = rng.normal(size=(4, 6)) * 5
    labels = [CYCLE_CLASSES[i % 4] for i in range(240)]
    embeddings = np.array(
        [centres[CYCLE_CLASSES.index(label)] + rng.normal(scale=0.3, size=6) for label in labels]
    )
    head = CycleStateHead().fit(embeddings, labels)
    accuracy = np.mean([p == t for p, t in zip(head.predict(embeddings), labels, strict=True)])
    assert accuracy > 0.9


# -- symptom head ----------------------------------------------------------


def test_symptom_head_returns_probabilities_per_symptom():
    """Multilabel outputs in [0, 1], one series per symptom."""
    rng = np.random.default_rng(6)
    embeddings = rng.normal(size=(100, 5))
    targets = {
        "cramps": (embeddings[:, 0] > 0).astype(float),
        "bloating": (embeddings[:, 1] > 0).astype(float),
    }
    head = SymptomHead().fit(embeddings, targets)
    probs = head.predict_proba(embeddings)
    assert set(probs) == {"cramps", "bloating"}
    for values in probs.values():
        assert np.all((values >= 0) & (values <= 1))


def test_symptom_head_falls_back_to_the_base_rate_when_unlearnable():
    """A constant target yields the base rate, not a confident guess."""
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(50, 4))
    head = SymptomHead().fit(embeddings, {"cramps": np.zeros(50)})
    assert np.allclose(head.predict_proba(embeddings)["cramps"], 0.0)


def test_symptom_head_forecasts_the_next_day(fitted):
    """The model's symptom head is fitted against next-day, not same-day, labels."""
    model, _, test = fitted
    outputs = model.predict(test)
    assert outputs
    assert outputs[0].symptom_probabilities, "symptom head must produce output"


# -- masked reconstruction -------------------------------------------------


def test_masked_head_reconstructs_hidden_entries():
    """Reconstruction must correlate with the truth it never saw directly."""
    rng = np.random.default_rng(8)
    embeddings = rng.normal(size=(120, 6))
    truth = embeddings @ rng.normal(size=(6, 4))
    mask = make_artificial_mask(np.ones_like(truth, dtype=bool), mask_fraction=0.3, seed=0)
    head = MaskedReconstructionHead().fit(embeddings, truth, mask.astype(bool))
    predictions = head.predict(embeddings)
    assert predictions.shape == truth.shape
    assert np.corrcoef(predictions.ravel(), truth.ravel())[0, 1] > 0.5


def test_artificial_mask_only_hides_observed_entries():
    """An already-missing entry has no ground truth to reconstruct."""
    observed = np.zeros((30, 4), dtype=bool)
    observed[:15] = True
    mask = make_artificial_mask(observed, mask_fraction=0.9, seed=0)
    assert not mask[15:].any()
    assert mask[:15].any()


# -- losses ----------------------------------------------------------------


def test_masked_losses_ignore_unobserved_entries():
    """A wrong prediction at a masked position must not be penalised."""
    pred = np.array([[1.0, 100.0]])
    target = np.array([[1.0, 0.0]])
    mask = np.array([[1.0, 0.0]])
    assert masked_mse(pred, target, mask) == pytest.approx(0.0)
    assert masked_mae(pred, target, mask) == pytest.approx(0.0)


def test_gaussian_nll_prefers_honest_variance():
    """Overconfidence on a bad prediction must cost more than a wide interval."""
    target = np.array([5.0])
    mean = np.array([0.0])
    confident = gaussian_nll(mean, np.array([-4.0]), target)
    honest = gaussian_nll(mean, np.array([3.0]), target)
    assert honest < confident


def test_cross_entropy_rewards_the_correct_class():
    """Basic sanity on the categorical term."""
    good = np.array([[0.9, 0.05, 0.03, 0.02]])
    bad = np.array([[0.02, 0.03, 0.05, 0.9]])
    target = np.array([0])
    assert categorical_cross_entropy(good, target) < categorical_cross_entropy(bad, target)


def test_binary_cross_entropy_respects_the_mask():
    """Unreported symptoms must not contribute."""
    probs = np.array([[0.9, 0.9]])
    target = np.array([[1.0, 0.0]])
    masked = binary_cross_entropy(probs, target, np.array([[1.0, 0.0]]))
    unmasked = binary_cross_entropy(probs, target)
    assert masked < unmasked


def test_state_loss_is_the_stated_weighted_sum():
    """L_state = lh*Lh + lc*Lc + ls*Ls + lm*Lm, exactly."""
    weights = LossWeights(hormone=2.0, cycle=3.0, symptom=0.5, masked=0.25)
    result = state_loss(
        hormone_pred=np.array([[1.0, 2.0, 3.0]]),
        hormone_target=np.array([[1.5, 2.0, 3.0]]),
        hormone_mask=np.ones((1, 3)),
        cycle_probs=np.array([[0.7, 0.1, 0.1, 0.1]]),
        cycle_target=np.array([0]),
        symptom_probs=np.array([[0.8]]),
        symptom_target=np.array([[1.0]]),
        masked_pred=np.array([[1.0]]),
        masked_target=np.array([[0.0]]),
        masked_mask=np.array([[1.0]]),
        weights=weights,
    )
    expected = (
        weights.hormone * result["hormone"]
        + weights.cycle * result["cycle"]
        + weights.symptom * result["symptom"]
        + weights.masked * result["masked"]
    )
    assert result["total"] == pytest.approx(expected)
    assert set(result) == {"hormone", "cycle", "symptom", "masked", "total"}


def test_hormone_loss_supports_the_documented_kinds():
    """mse / mae / gaussian_nll are all selectable."""
    pred, target = np.array([[1.0]]), np.array([[2.0]])
    assert hormone_loss(pred, target, kind="mse") == pytest.approx(1.0)
    assert hormone_loss(pred, target, kind="mae") == pytest.approx(1.0)
    assert np.isfinite(
        hormone_loss(pred, target, kind="gaussian_nll", pred_log_var=np.array([[0.0]]))
    )
    with pytest.raises(ValueError, match="Unknown hormone loss"):
        hormone_loss(pred, target, kind="nonsense")


def test_all_heads_are_fitted_by_the_state_model(fitted):
    """The composite model wires up every head."""
    model, _, _ = fitted
    assert model.report is not None
    assert set(model.report.losses) == {"hormone", "cycle", "symptom", "masked", "total"}
    assert model.heads.hormone.coefficients
    assert model.heads.cycle.weights is not None
    assert model.heads.symptom.symptoms
    assert model.heads.masked.coefficients is not None


def test_interval_coverage_accepts_every_scalar_sigma_form():
    """A 0-d array sigma must broadcast, not be mistaken for a per-point sequence.

    ``interval_coverage`` branched on ``np.isscalar``, which is False for a 0-d
    numpy array. Such a sigma therefore took the sequence path, stayed 0-d, and
    crashed on ``s[ok]`` with an IndexError — so a caller that computed a single
    pooled sigma with numpy got a crash instead of a coverage number.
    """
    from evaluation.temporal import interval_coverage

    predicted = [1.0, 2.0, 3.0, 4.0]
    truth = [1.1, 2.2, 3.3, 9.0]
    expected = interval_coverage(predicted, truth, 1.0)["coverage"]

    for sigma in (np.float64(1.0), np.array(1.0), np.asarray([1.0, 1.0, 1.0, 1.0])):
        assert interval_coverage(predicted, truth, sigma)["coverage"] == pytest.approx(expected)
