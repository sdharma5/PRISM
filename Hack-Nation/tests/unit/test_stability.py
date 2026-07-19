"""Bootstrap Jaccard, flip rate, JS divergence and ablation sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.stability import (
    assignment_entropy,
    jaccard,
    jensen_shannon_divergence,
    match_clusters,
    subtype_flip_rate,
)
from models.phenotype.clustering import ClusteringInput, fit_clustering
from models.stability.ablation import run_ablation
from models.stability.bootstrap import align_labels, bootstrap_clustering
from models.stability.calibration import fit_temperature, membership_from_distances
from models.stability.perturbation import ASSAY_CV, apply_assay_noise, run_perturbations
from tests.fixtures.synthetic_clusters import make_synthetic_cluster_frame


def _input(n_per_group: int = 30, noise: float = 0.35, seed: int = 0):
    frame, truth = make_synthetic_cluster_frame(n_per_group=n_per_group, noise=noise, seed=seed)
    return (
        ClusteringInput(
            label="raw_standardized",
            matrix=frame.to_numpy(),
            participant_ids=list(frame.index),
            feature_names=list(frame.columns),
        ),
        truth,
    )


# ---------------------------------------------------------------- primitives


def test_jaccard_bounds():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert jaccard({1, 2}, {3, 4}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_match_clusters_is_invariant_to_label_permutation():
    reference = np.array([0, 0, 0, 1, 1, 1])
    permuted = np.array([9, 9, 9, 4, 4, 4])
    assert match_clusters(reference, permuted) == {0: 1.0, 1: 1.0}


def test_align_labels_undoes_permutation():
    reference = np.array([0, 0, 1, 1, 2, 2])
    candidate = np.array([2, 2, 0, 0, 1, 1])
    mapping = align_labels(reference, candidate)
    assert [mapping[int(c)] for c in candidate] == list(reference)


def test_subtype_flip_rate_counts_changes():
    assert subtype_flip_rate([0, 1, 2], [0, 1, 2]) == 0.0
    assert subtype_flip_rate([0, 1, 2], [1, 1, 2]) == pytest.approx(1 / 3)
    assert subtype_flip_rate([0, 1], [1, 0]) == 1.0
    with pytest.raises(ValueError):
        subtype_flip_rate([0, 1], [0])


def test_assignment_entropy_is_zero_when_certain_and_one_when_uniform():
    assert assignment_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert assignment_entropy([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(1.0)
    assert 0.0 < assignment_entropy([0.6, 0.3, 0.1]) < 1.0


def test_js_divergence_bounds_and_symmetry():
    assert jensen_shannon_divergence([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    assert jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-6)
    p, q = [0.7, 0.2, 0.1], [0.2, 0.7, 0.1]
    assert jensen_shannon_divergence(p, q) == pytest.approx(jensen_shannon_divergence(q, p))
    assert 0.0 < jensen_shannon_divergence(p, q) < 1.0


def test_js_divergence_accepts_dicts_with_differing_keys():
    value = jensen_shannon_divergence({"a": 1.0}, {"b": 1.0})
    assert value == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- bootstrap


def test_bootstrap_jaccard_is_high_for_well_separated_groups():
    data, _ = _input()
    fitted = fit_clustering(data, "kmeans", k=3, seed=0)
    result = bootstrap_clustering(data, "kmeans", 3, fitted.labels, n_bootstrap=15, seed=0)
    assert result.n_effective_resamples > 10
    assert result.mean_jaccard > 0.75
    assert result.mean_ari is not None and result.mean_ari > 0.8
    assert set(result.assignment_entropy) == set(data.participant_ids)
    assert np.mean(list(result.agreement_rate.values())) > 0.85


def test_bootstrap_jaccard_is_low_for_structureless_data():
    rng = np.random.default_rng(1)
    data = ClusteringInput(
        label="noise",
        matrix=rng.normal(size=(60, 5)),
        participant_ids=[f"p{i}" for i in range(60)],
        feature_names=[f"f{i}" for i in range(5)],
    )
    result = bootstrap_clustering(data, "kmeans", 4, n_bootstrap=15, seed=0)
    assert result.mean_jaccard < 0.75


def test_bootstrap_entropy_is_lower_for_clean_than_noisy_cohorts():
    clean, _ = _input(noise=0.25)
    messy, _ = _input(noise=2.2, seed=3)
    clean_entropy = np.mean(
        list(
            bootstrap_clustering(
                clean, "kmeans", 3, n_bootstrap=12, seed=0
            ).assignment_entropy.values()
        )
    )
    messy_entropy = np.mean(
        list(
            bootstrap_clustering(
                messy, "kmeans", 3, n_bootstrap=12, seed=0
            ).assignment_entropy.values()
        )
    )
    assert clean_entropy < messy_entropy


# ------------------------------------------------------------------ ablation


def test_ablation_reports_a_flip_rate_per_feature_and_a_fragility_winner():
    data, _ = _input()
    fitted = fit_clustering(data, "kmeans", k=3, seed=0)
    result = run_ablation(
        data,
        fitted.labels,
        "kmeans",
        3,
        modality_of={
            f: ("lab" if "insulin" in f or "testosterone" in f else "anthro")
            for f in data.feature_names
        },
    )
    assert set(result.flip_rate) == set(data.feature_names)
    assert all(0.0 <= v <= 1.0 for v in result.flip_rate.values())
    assert result.highest_fragility_feature in data.feature_names
    assert set(result.modality_flip_rate) == {"lab", "anthro"}


def test_ablation_flip_rate_is_low_for_a_robust_structure():
    data, _ = _input(n_per_group=40, noise=0.25)
    fitted = fit_clustering(data, "kmeans", k=3, seed=0)
    result = run_ablation(data, fitted.labels, "kmeans", 3)
    assert np.mean(list(result.flip_rate.values())) < 0.25


# --------------------------------------------------------------- perturbation


def test_assay_cv_table_is_documented_and_plausible():
    assert "total_testosterone" in ASSAY_CV
    assert all(0.0 < cv < 0.5 for cv in ASSAY_CV.values())
    # Immunoassay steroids are noisier than automated chemistry.
    assert ASSAY_CV["total_testosterone"] > ASSAY_CV["fasting_glucose"]


def test_assay_noise_perturbs_without_destroying_the_matrix():
    data, _ = _input()
    rng = np.random.default_rng(0)
    noisy = apply_assay_noise(data.matrix, data.feature_names, rng)
    assert noisy.shape == data.matrix.shape
    assert not np.allclose(noisy, data.matrix)
    assert np.abs(noisy - data.matrix).mean() < 1.0


def test_perturbations_report_flip_rate_and_js_divergence():
    data, _ = _input()
    fitted = fit_clustering(data, "kmeans", k=3, seed=0)
    results = run_perturbations(
        data,
        fitted.labels,
        "kmeans",
        3,
        n_noise_replicates=3,
        scaling_strategies=("robust",),
        imputation_strategies=("median",),
    )
    assert {r.scenario for r in results} >= {"scaling_robust", "imputation_median"}
    for result in results:
        assert 0.0 <= result.flip_rate <= 1.0
        assert 0.0 <= result.mean_js_divergence <= 1.0
        assert set(result.per_participant_flipped) == set(data.participant_ids)
    noise_flips = [r.flip_rate for r in results if r.scenario.startswith("assay_noise")]
    assert max(noise_flips) < 0.2


# --------------------------------------------------------------- calibration


def test_temperature_scaling_preserves_the_argmax():
    data, _ = _input()
    fitted = fit_clustering(data, "kmeans", k=3, seed=0)
    boot = bootstrap_clustering(data, "kmeans", 3, fitted.labels, n_bootstrap=10, seed=0)
    raw = membership_from_distances(data.matrix, np.asarray(fitted.centers))
    agreement = [boot.agreement_rate[pid] for pid in data.participant_ids]
    result = fit_temperature(raw, agreement)
    from models.stability.calibration import temperature_scale

    scaled = temperature_scale(raw, result.temperature)
    assert (scaled.argmax(axis=1) == raw.argmax(axis=1)).all()
    assert result.ece_after <= result.ece_before + 1e-9
    assert result.temperature > 0
