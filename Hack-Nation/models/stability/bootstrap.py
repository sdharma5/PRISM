"""Bootstrap resampling of a clustering solution.

Scientific WHY
--------------
If a discovered grouping only exists in the exact cohort we happened to sample,
it is a property of the sample, not of the biology. Bootstrap resampling
approximates "what if we had recruited a slightly different cohort?" and asks
whether the same groups reappear.

We report three complementary things:

* **Per-cluster Jaccard** (Hennig 2007): some clusters in a solution are solid
  while others dissolve on resampling. A single global number hides that, so we
  keep the per-cluster values and let the abstention layer look at the cluster
  the *individual patient* was placed in.
* **ARI across bootstrap runs**: whole-partition reproducibility.
* **Per-participant assignment entropy**: how often *this* participant landed in
  the same group across resamples. This is the patient-level number the report
  ultimately needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.clustering import adjusted_rand
from evaluation.stability import assignment_entropy, bootstrap_jaccard, jaccard
from models.phenotype.clustering import ClusteringInput, fit_base_partition

__all__ = ["BootstrapResult", "align_labels", "bootstrap_clustering"]


def align_labels(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[int, int]:
    """Map candidate cluster ids onto reference cluster ids by maximum overlap.

    Cluster labels are arbitrary integers: a re-fit may call "the metabolic-
    leaning group" 0 instead of 2. Without alignment every stability statistic
    would be dominated by label permutation rather than by real reassignment.
    We solve the assignment problem exactly (Hungarian algorithm) on the Jaccard
    overlap matrix; unmatched candidate clusters map to -1.
    """
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    ref_labels = list(np.unique(reference))
    cand_labels = list(np.unique(candidate))

    overlap = np.zeros((len(cand_labels), len(ref_labels)), dtype=float)
    for i, cl in enumerate(cand_labels):
        cand_members = set(np.flatnonzero(candidate == cl).tolist())
        for j, rl in enumerate(ref_labels):
            ref_members = set(np.flatnonzero(reference == rl).tolist())
            overlap[i, j] = jaccard(cand_members, ref_members)

    mapping = {int(cl): -1 for cl in cand_labels}
    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)
        for i, j in zip(rows, cols, strict=True):
            if overlap[i, j] > 0:
                mapping[int(cand_labels[i])] = int(ref_labels[j])
    return mapping


@dataclass
class BootstrapResult:
    """Everything a stability report needs from a bootstrap sweep."""

    reference_labels: np.ndarray
    participant_ids: list[str]
    per_cluster_jaccard: dict[str, float] = field(default_factory=dict)
    mean_jaccard: float = 0.0
    mean_ari: float | None = None
    #: participant_id -> {reference_cluster_id: relative frequency}
    assignment_distribution: dict[str, dict[int, float]] = field(default_factory=dict)
    #: participant_id -> normalized Shannon entropy of that distribution
    assignment_entropy: dict[str, float] = field(default_factory=dict)
    #: participant_id -> fraction of resamples agreeing with the reference label
    agreement_rate: dict[str, float] = field(default_factory=dict)
    n_effective_resamples: int = 0
    warnings: list[str] = field(default_factory=list)

    def cluster_jaccard_for(self, cluster: int) -> float | None:
        return self.per_cluster_jaccard.get(f"cluster_{int(cluster)}")


def bootstrap_clustering(
    data: ClusteringInput,
    algorithm: str,
    k: int,
    reference_labels: np.ndarray | None = None,
    n_bootstrap: int = 50,
    seed: int = 0,
) -> BootstrapResult:
    """Resample participants with replacement, re-cluster, and score stability.

    Consensus clustering is bootstrapped through its k-means base learner: the
    consensus step is itself a resampling procedure, and nesting one inside the
    other costs an order of magnitude of compute for no extra information.
    """
    X = np.asarray(data.matrix, dtype=float)
    n = X.shape[0]
    base = "kmeans" if algorithm == "consensus" else algorithm

    if reference_labels is None:
        reference_labels = np.asarray(fit_base_partition(X, base, k, seed))
    reference_labels = np.asarray(reference_labels)

    rng = np.random.default_rng(seed)
    runs: list[tuple[np.ndarray, list[int]]] = []
    votes: dict[int, dict[int, int]] = {i: {} for i in range(n)}
    aris: list[float] = []
    warnings: list[str] = []

    for b in range(n_bootstrap):
        draw = np.asarray(rng.choice(n, size=n, replace=True))
        idx = sorted({int(i) for i in draw})
        if len(idx) <= k:
            continue
        sub = X[idx]
        try:
            labels = np.asarray(fit_base_partition(sub, base, k, seed=seed + b + 1))
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.append(f"bootstrap {b} failed: {exc}")
            continue

        runs.append((labels, idx))
        ref_sub = reference_labels[idx]
        aris.append(adjusted_rand(ref_sub, labels))

        mapping = align_labels(ref_sub, labels)
        for position, original_index in enumerate(idx):
            mapped = mapping.get(int(labels[position]), -1)
            if mapped < 0:
                continue
            votes[original_index][mapped] = votes[original_index].get(mapped, 0) + 1

    per_cluster = bootstrap_jaccard(reference_labels, runs) if runs else {"mean": 0.0}
    mean_jaccard = float(per_cluster.get("mean", 0.0))

    distribution: dict[str, dict[int, float]] = {}
    entropies: dict[str, float] = {}
    agreement: dict[str, float] = {}
    for i, pid in enumerate(data.participant_ids):
        counts = votes[i]
        total = sum(counts.values())
        if total == 0:
            distribution[pid] = {}
            entropies[pid] = 1.0
            agreement[pid] = 0.0
            continue
        dist = {int(c): v / total for c, v in counts.items()}
        distribution[pid] = dist
        entropies[pid] = assignment_entropy(list(dist.values()))
        agreement[pid] = dist.get(int(reference_labels[i]), 0.0)

    if not runs:
        warnings.append("no bootstrap resample produced a usable partition")

    return BootstrapResult(
        reference_labels=reference_labels,
        participant_ids=list(data.participant_ids),
        per_cluster_jaccard={k_: v for k_, v in per_cluster.items() if k_ != "mean"},
        mean_jaccard=mean_jaccard,
        mean_ari=float(np.mean(aris)) if aris else None,
        assignment_distribution=distribution,
        assignment_entropy=entropies,
        agreement_rate=agreement,
        n_effective_resamples=len(runs),
        warnings=warnings,
    )
