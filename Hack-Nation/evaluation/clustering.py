"""Internal- and external-validity metrics for unsupervised phenotype discovery.

Scientific WHY
--------------
Unsupervised subtype discovery has no ground truth, so a single index can always
be gamed by the choice of representation or K. We therefore report a *panel*:

* Silhouette (higher better) — geometric separation, biased toward spherical,
  equal-size clusters, so it should never be read alone.
* Calinski-Harabasz (higher better) — variance-ratio criterion; grows with K for
  many data geometries, which is exactly why we do not select K by it alone.
* Davies-Bouldin (lower better) — average worst-case cluster-pair similarity.
* ARI / NMI across seeds (higher better) — *reproducibility*, not separation. A
  partition that no two random restarts agree on is not a finding.

Every wrapper returns ``None`` rather than raising when the metric is undefined
(e.g. a degenerate single-cluster partition), because a benchmark sweep must be
able to record "this configuration collapsed" instead of aborting.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

__all__ = [
    "adjusted_rand",
    "calinski_harabasz",
    "davies_bouldin",
    "normalized_mutual_info",
    "pairwise_agreement",
    "silhouette",
    "valid_partition",
]


def valid_partition(labels: np.ndarray) -> bool:
    """True when ``labels`` contains at least two clusters and fewer than n.

    Internal indices are undefined for a partition with one cluster (no
    between-cluster scatter) or with every point in its own cluster.
    """
    labels = np.asarray(labels)
    n_unique = len(np.unique(labels))
    return 1 < n_unique < len(labels)


def silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean silhouette coefficient, or ``None`` for a degenerate partition."""
    if not valid_partition(labels):
        return None
    try:
        return float(silhouette_score(X, labels))
    except ValueError:
        return None


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Variance-ratio criterion, or ``None`` for a degenerate partition."""
    if not valid_partition(labels):
        return None
    try:
        return float(calinski_harabasz_score(X, labels))
    except ValueError:
        return None


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Davies-Bouldin index (lower is better), or ``None`` if undefined."""
    if not valid_partition(labels):
        return None
    try:
        return float(davies_bouldin_score(X, labels))
    except ValueError:
        return None


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index between two labellings of the same samples."""
    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mutual information between two labellings."""
    return float(normalized_mutual_info_score(np.asarray(a), np.asarray(b)))


def pairwise_agreement(labelings: list[np.ndarray], metric: str = "ari") -> float | None:
    """Mean pairwise ARI (or NMI) over every pair of labellings.

    This is our reproducibility statistic: it answers "if I rerun this with a
    different random seed, do I recover the same partition?".
    """
    if len(labelings) < 2:
        return None
    fn = adjusted_rand if metric == "ari" else normalized_mutual_info
    scores = [
        fn(labelings[i], labelings[j])
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    return float(np.mean(scores)) if scores else None
