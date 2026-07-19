"""The benchmark must recover planted groups and choose K from evidence."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.clustering import adjusted_rand
from models.phenotype.clustering import (
    ClusteringInput,
    consensus_matrix,
    fit_clustering,
    run_clustering_benchmark,
    select_k,
)
from tests.fixtures.synthetic_clusters import make_synthetic_cluster_frame


def _input(
    seed: int = 0, n_per_group: int = 30, groups=("metabolic_like", "androgen_like", "lh_amh_like")
):
    frame, labels = make_synthetic_cluster_frame(n_per_group=n_per_group, groups=groups, seed=seed)
    data = ClusteringInput(
        label="raw_standardized",
        matrix=frame.to_numpy(),
        participant_ids=list(frame.index),
        feature_names=list(frame.columns),
    )
    return data, labels


def test_recovers_three_planted_groups():
    data, truth = _input()
    fitted = fit_clustering(data, "kmeans", k=3, seed=0)
    codes = {name: i for i, name in enumerate(dict.fromkeys(truth))}
    assert adjusted_rand([codes[t] for t in truth], fitted.labels) > 0.9


@pytest.mark.parametrize("algorithm", ["kmeans", "gaussian_mixture", "agglomerative", "consensus"])
def test_every_algorithm_recovers_the_planted_structure(algorithm):
    data, truth = _input()
    fitted = fit_clustering(data, algorithm, k=3, seed=0, consensus_resamples=15)
    codes = {name: i for i, name in enumerate(dict.fromkeys(truth))}
    assert adjusted_rand([codes[t] for t in truth], fitted.labels) > 0.8


def test_consensus_matrix_is_symmetric_with_unit_diagonal():
    data, _ = _input()
    matrix = consensus_matrix(data.matrix, k=3, n_resamples=10, seed=0)
    assert matrix.shape == (data.matrix.shape[0], data.matrix.shape[0])
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_benchmark_refuses_to_cluster_everyone_by_default():
    data, _ = _input()
    with pytest.raises(ValueError, match="cluster_subset_ids is required"):
        run_clustering_benchmark([data], cluster_subset_ids=[])


def test_benchmark_only_clusters_the_supplied_subset():
    data, _ = _input()
    subset = data.participant_ids[:45]
    results = run_clustering_benchmark(
        [data],
        cluster_subset_ids=subset,
        algorithms=("kmeans",),
        k_values=(2, 3),
        seeds=(0, 1),
        n_bootstrap=5,
    )
    assert results
    assert all(r.n_samples == 45 for r in results)


def test_benchmark_covers_the_full_cross_product():
    data, _ = _input()
    algorithms = ("kmeans", "agglomerative")
    k_values = (2, 3, 4)
    results = run_clustering_benchmark(
        [data],
        cluster_subset_ids=data.participant_ids,
        algorithms=algorithms,
        k_values=k_values,
        seeds=(0, 1),
        n_bootstrap=5,
    )
    assert len(results) == len(algorithms) * len(k_values)
    assert {(r.algorithm, r.k) for r in results} == {(a, k) for a in algorithms for k in k_values}


def test_select_k_is_evidence_based_and_recovers_the_planted_k():
    """Three groups were planted, so evidence — not a K=4 prior — should win."""
    data, _ = _input(n_per_group=35)
    results = run_clustering_benchmark(
        [data],
        cluster_subset_ids=data.participant_ids,
        algorithms=("kmeans", "agglomerative"),
        k_values=(2, 3, 4, 5, 6),
        seeds=(0, 1, 2),
        n_bootstrap=10,
    )
    selection = select_k(results)
    assert selection.k == 3
    assert "No prior favoured any particular K" in selection.rationale
    assert len(selection.ranked) == len(results)


def test_select_k_finds_two_groups_when_only_two_were_planted():
    """K selection must track the data, not a fixed publication-derived number."""
    data, _ = _input(n_per_group=40, groups=("metabolic_like", "lh_amh_like"))
    results = run_clustering_benchmark(
        [data],
        cluster_subset_ids=data.participant_ids,
        algorithms=("kmeans",),
        k_values=(2, 3, 4),
        seeds=(0, 1, 2),
        n_bootstrap=10,
    )
    assert select_k(results).k == 2


def test_select_k_docstring_documents_the_no_default_four_rule():
    doc = select_k.__doc__ or ""
    assert "four clusters is never" in doc
    assert "K=4" in doc


def test_select_k_warns_when_nothing_is_stable():
    """Unstructured noise must not be reported as a clean solution."""
    rng = np.random.default_rng(0)
    noise = ClusteringInput(
        label="pure_noise",
        matrix=rng.normal(size=(60, 6)),
        participant_ids=[f"p{i}" for i in range(60)],
        feature_names=[f"f{i}" for i in range(6)],
    )
    results = run_clustering_benchmark(
        [noise],
        cluster_subset_ids=noise.participant_ids,
        algorithms=("kmeans",),
        k_values=(2, 3, 4),
        seeds=(0, 1, 2),
        n_bootstrap=10,
    )
    assert select_k(results).warnings


def test_multiple_representations_are_labelled_separately():
    data, _ = _input()
    subset = ClusteringInput(
        label="named_subset",
        matrix=data.matrix[:, :3],
        participant_ids=data.participant_ids,
        feature_names=data.feature_names[:3],
    )
    results = run_clustering_benchmark(
        [data, subset],
        cluster_subset_ids=data.participant_ids,
        algorithms=("kmeans",),
        k_values=(3,),
        seeds=(0,),
        n_bootstrap=3,
    )
    assert {r.representation for r in results} == {"raw_standardized", "named_subset"}
