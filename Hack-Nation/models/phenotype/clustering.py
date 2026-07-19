"""Clustering benchmark over (representation x algorithm x K) for phenotype discovery.

Scientific WHY
--------------
Reported hormonal-health "subtypes" are extremely sensitive to three choices that
papers often fix silently: which *representation* is clustered (raw standardized
labs, interpretable domain scores, a learned embedding, a curated subset), which
*algorithm* is used, and which *K* is assumed. This module makes all three an
explicit, logged sweep so that a result can be judged on whether it survives the
sweep rather than on whether it was produced at all.

We also include **consensus clustering** (Monti et al. 2003): cluster many
subsamples with many seeds, accumulate a co-association matrix of how often each
pair of participants lands together, and cut the resulting similarity structure
hierarchically. Consensus partitions are markedly less seed-dependent, and the
co-association matrix itself is a stability diagnostic.

Design constraints enforced here
--------------------------------
* Clustering is only ever run on a **caller-supplied subset** of participants
  (e.g. the PCOS-positive training split). ``run_clustering_benchmark`` raises if
  the subset is not given, so nobody can accidentally discover "subtypes" over a
  mixed case/control cohort and then interpret the case/control axis as biology.
* No PCOS-specific logic lives here. This module knows about matrices only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from evaluation.clustering import (
    calinski_harabasz,
    davies_bouldin,
    pairwise_agreement,
    silhouette,
)
from evaluation.stability import bootstrap_jaccard
from schemas.phenotype import ClusteringBenchmark

__all__ = [
    "ALGORITHMS",
    "ClusteringInput",
    "FittedClustering",
    "KSelection",
    "consensus_matrix",
    "fit_base_partition",
    "fit_clustering",
    "run_clustering_benchmark",
    "select_k",
]

Algorithm = Literal["kmeans", "gaussian_mixture", "agglomerative", "spectral", "consensus"]

#: Algorithms enabled by default. ``spectral`` is optional (it is O(n^3) in the
#: affinity decomposition and unstable on small cohorts) and must be opted into.
ALGORITHMS: tuple[str, ...] = ("kmeans", "gaussian_mixture", "agglomerative", "consensus")

DEFAULT_K_RANGE: tuple[int, ...] = (2, 3, 4, 5, 6)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class ClusteringInput:
    """One representation to be clustered.

    Accepting an arbitrary matrix plus a label is deliberate: it keeps this module
    independent of the autoencoder / domain-score modules, so an embedding can be
    benchmarked without importing (or requiring) torch.
    """

    label: str
    matrix: np.ndarray
    participant_ids: list[str]
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(f"{self.label}: matrix must be 2-D, got shape {self.matrix.shape}.")
        if len(self.participant_ids) != self.matrix.shape[0]:
            raise ValueError(
                f"{self.label}: {len(self.participant_ids)} ids vs {self.matrix.shape[0]} rows."
            )

    def subset(self, ids: Sequence[str]) -> ClusteringInput:
        """Restrict to ``ids``, preserving their order in this representation."""
        wanted = set(ids)
        keep = [i for i, pid in enumerate(self.participant_ids) if pid in wanted]
        if not keep:
            raise ValueError(f"{self.label}: none of the requested participants are present.")
        return ClusteringInput(
            label=self.label,
            matrix=self.matrix[keep, :],
            participant_ids=[self.participant_ids[i] for i in keep],
            feature_names=list(self.feature_names),
        )


@dataclass
class FittedClustering:
    """A fitted partition plus everything downstream stability code needs."""

    representation: str
    algorithm: str
    k: int
    seed: int
    labels: np.ndarray
    centers: np.ndarray | None = None
    responsibilities: np.ndarray | None = None
    participant_ids: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    model: Any = None

    @property
    def n_clusters_found(self) -> int:
        return int(len(np.unique(self.labels)))


def _cluster_centers(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Empirical centroid per label; used for algorithms with no native centers."""
    uniq = np.unique(labels)
    return np.vstack([X[labels == label].mean(axis=0) for label in uniq])


def consensus_matrix(
    X: np.ndarray,
    k: int,
    n_resamples: int = 40,
    subsample_fraction: float = 0.8,
    seed: int = 0,
    base_algorithm: str = "kmeans",
) -> np.ndarray:
    """Co-association matrix: P(i and j co-cluster | both sampled).

    Entry (i, j) is the number of resamples in which i and j were assigned the
    same cluster, divided by the number of resamples in which both were drawn.
    Pairs never co-drawn get 0.0 and are treated as maximally dissimilar.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    together = np.zeros((n, n), dtype=float)
    co_drawn = np.zeros((n, n), dtype=float)
    size = max(k + 1, int(round(subsample_fraction * n)))

    for r in range(n_resamples):
        idx = rng.choice(n, size=min(size, n), replace=False)
        sub = X[idx]
        if len(np.unique(sub, axis=0)) < k:
            continue
        labels = _fit_base(sub, base_algorithm, k, seed=int(seed * 1000 + r))
        co_drawn[np.ix_(idx, idx)] += 1.0
        for label in np.unique(labels):
            members = idx[labels == label]
            together[np.ix_(members, members)] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = np.where(co_drawn > 0, together / np.maximum(co_drawn, 1.0), 0.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def _fit_base(X: np.ndarray, algorithm: str, k: int, seed: int) -> np.ndarray:
    """Fit one non-consensus algorithm and return integer labels."""
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    if algorithm == "gaussian_mixture":
        return GaussianMixture(
            n_components=k, random_state=seed, covariance_type="full", n_init=3, reg_covar=1e-4
        ).fit_predict(X)
    if algorithm == "agglomerative":
        # Ward linkage minimizes within-cluster variance, matching the k-means
        # objective while producing a deterministic, seed-independent tree.
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if algorithm == "spectral":
        return SpectralClustering(
            n_clusters=k, random_state=seed, affinity="nearest_neighbors", assign_labels="kmeans"
        ).fit_predict(X)
    raise ValueError(f"Unknown algorithm '{algorithm}'.")


#: Public alias: stability modules re-fit partitions on perturbed matrices and
#: must use exactly the same base learner as the benchmark did.
fit_base_partition = _fit_base


def fit_clustering(
    data: ClusteringInput,
    algorithm: str,
    k: int,
    seed: int = 0,
    consensus_resamples: int = 40,
) -> FittedClustering:
    """Fit one (algorithm, K, seed) configuration on one representation."""
    X = np.asarray(data.matrix, dtype=float)
    if X.shape[0] <= k:
        raise ValueError(f"Cannot fit K={k} on {X.shape[0]} participants.")

    responsibilities: np.ndarray | None = None
    model: Any = None

    if algorithm == "consensus":
        coassoc = consensus_matrix(X, k=k, n_resamples=consensus_resamples, seed=seed)
        distance = 1.0 - coassoc
        np.fill_diagonal(distance, 0.0)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(distance)
    elif algorithm == "gaussian_mixture":
        model = GaussianMixture(
            n_components=k, random_state=seed, covariance_type="full", n_init=3, reg_covar=1e-4
        ).fit(X)
        labels = model.predict(X)
        responsibilities = model.predict_proba(X)
    elif algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        labels = model.labels_
    else:
        labels = _fit_base(X, algorithm, k, seed)

    centers = getattr(model, "cluster_centers_", None)
    if centers is None:
        means = getattr(model, "means_", None)
        centers = means if means is not None else _cluster_centers(X, labels)

    return FittedClustering(
        representation=data.label,
        algorithm=algorithm,
        k=k,
        seed=seed,
        labels=np.asarray(labels),
        centers=np.asarray(centers, dtype=float),
        responsibilities=responsibilities,
        participant_ids=list(data.participant_ids),
        feature_names=list(data.feature_names),
        model=model,
    )


def _bootstrap_stability(
    data: ClusteringInput,
    algorithm: str,
    k: int,
    reference_labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> float | None:
    """Mean per-cluster bootstrap Jaccard for one configuration."""
    if n_bootstrap <= 0:
        return None
    X = np.asarray(data.matrix, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    runs: list[tuple[np.ndarray, list[int]]] = []
    for b in range(n_bootstrap):
        idx = np.asarray(rng.choice(n, size=n, replace=True))
        unique_idx = sorted({int(i) for i in idx})
        if len(unique_idx) <= k:
            continue
        sub = X[unique_idx]
        try:
            base = "kmeans" if algorithm == "consensus" else algorithm
            labels = _fit_base(sub, base, k, seed + b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        runs.append((np.asarray(labels), unique_idx))
    if not runs:
        return None
    return bootstrap_jaccard(reference_labels, runs)["mean"]


def run_clustering_benchmark(
    representations: Sequence[ClusteringInput],
    cluster_subset_ids: Sequence[str],
    algorithms: Sequence[str] = ALGORITHMS,
    k_values: Sequence[int] = DEFAULT_K_RANGE,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_bootstrap: int = 20,
    consensus_resamples: int = 40,
) -> list[ClusteringBenchmark]:
    """Benchmark every (representation, algorithm, K) configuration.

    Parameters
    ----------
    cluster_subset_ids:
        **Required.** The participants to cluster — for the PCOS adapter this is
        the PCOS-positive *training* split. Passing an empty subset raises rather
        than defaulting to "everyone": clustering a mixed cohort and reading the
        dominant case/control axis as a phenotype axis is the single easiest way
        to manufacture a spurious subtype.

    Returns one :class:`ClusteringBenchmark` per configuration, with metrics
    computed on the first seed's partition and reproducibility (ARI/NMI)
    aggregated across all seeds.
    """
    if not cluster_subset_ids:
        raise ValueError(
            "cluster_subset_ids is required: clustering must be restricted to an "
            "explicit participant subset (e.g. PCOS-positive training participants)."
        )

    results: list[ClusteringBenchmark] = []
    for representation in representations:
        data = representation.subset(cluster_subset_ids)
        X = np.asarray(data.matrix, dtype=float)
        for algorithm in algorithms:
            for k in k_values:
                warnings: list[str] = []
                if X.shape[0] <= k:
                    results.append(
                        ClusteringBenchmark(
                            representation=data.label,
                            algorithm=algorithm,
                            k=k,
                            seed=int(seeds[0]),
                            n_samples=int(X.shape[0]),
                            warnings=[f"skipped: only {X.shape[0]} participants for K={k}"],
                        )
                    )
                    continue

                labelings: list[np.ndarray] = []
                primary: FittedClustering | None = None
                for seed in seeds:
                    try:
                        fitted = fit_clustering(
                            data,
                            algorithm,
                            k,
                            seed=int(seed),
                            consensus_resamples=consensus_resamples,
                        )
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        warnings.append(f"seed {seed} failed: {exc}")
                        continue
                    labelings.append(fitted.labels)
                    if primary is None:
                        primary = fitted

                if primary is None:
                    results.append(
                        ClusteringBenchmark(
                            representation=data.label,
                            algorithm=algorithm,
                            k=k,
                            seed=int(seeds[0]),
                            n_samples=int(X.shape[0]),
                            warnings=warnings or ["all seeds failed"],
                        )
                    )
                    continue

                if primary.n_clusters_found < k:
                    warnings.append(
                        f"collapsed: requested K={k}, recovered {primary.n_clusters_found}"
                    )

                results.append(
                    ClusteringBenchmark(
                        representation=data.label,
                        algorithm=algorithm,
                        k=k,
                        seed=int(primary.seed),
                        silhouette=silhouette(X, primary.labels),
                        calinski_harabasz=calinski_harabasz(X, primary.labels),
                        davies_bouldin=davies_bouldin(X, primary.labels),
                        mean_bootstrap_jaccard=_bootstrap_stability(
                            data, algorithm, k, primary.labels, n_bootstrap, seed=int(seeds[0])
                        ),
                        mean_ari_across_seeds=pairwise_agreement(labelings, "ari"),
                        mean_nmi_across_seeds=pairwise_agreement(labelings, "nmi"),
                        n_samples=int(X.shape[0]),
                        warnings=warnings,
                    )
                )
    return results


@dataclass(frozen=True)
class KSelection:
    """The chosen configuration plus the evidence and the runners-up."""

    k: int
    representation: str
    algorithm: str
    score: float
    ranked: list[tuple[str, str, int, float]]
    rationale: str
    warnings: list[str] = field(default_factory=list)


def _minmax(values: list[float | None], invert: bool = False) -> list[float]:
    """Scale a metric column to [0, 1]; missing values score 0.0."""
    present = [v for v in values if v is not None and np.isfinite(v)]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    span = hi - lo
    out: list[float] = []
    for v in values:
        if v is None or not np.isfinite(v):
            out.append(0.0)
        elif span <= 0:
            out.append(0.5)
        else:
            scaled = (v - lo) / span
            out.append(1.0 - scaled if invert else scaled)
    return out


def select_k(
    benchmarks: Sequence[ClusteringBenchmark],
    weights: dict[str, float] | None = None,
    min_bootstrap_jaccard: float = 0.6,
    min_ari: float = 0.5,
    min_silhouette: float = 0.25,
) -> KSelection:
    """Rank configurations on evidence and return the best-supported one.

    **K is never fixed a priori — and in particular, four clusters is never
    chosen by default just because the published PCOS literature commonly
    reports four subtypes (the four Rotterdam phenotype combinations, and the
    reproductive/metabolic groupings of Dapas et al. 2020).** Those are
    *hypotheses to be tested against our own data*, not a prior we encode. This
    function has no term that references K=4, and K enters the score only
    through the measured metrics. If our data support K=2 or K=5, that is what
    is returned; if the winning configuration happens to be K=4, the rationale
    string records that it won on measured separation and reproducibility.

    Scoring combines, after min-max scaling across the whole sweep:

    * ``silhouette`` (+), ``calinski_harabasz`` (+), ``davies_bouldin`` (−)
      — separation;
    * ``mean_bootstrap_jaccard`` (+), ``mean_ari_across_seeds`` (+) — is the
      partition reproducible at all;

    with reproducibility weighted at least as heavily as separation, because an
    unreproducible but well-separated partition is an artifact of one seed. A
    configuration failing ``min_bootstrap_jaccard``, ``min_ari`` or
    ``min_silhouette`` is retained in the ranking but flagged, so a reader can see
    whether *any* configuration cleared the bar — if none did, the honest
    conclusion is that this cohort does not support discrete subtypes at all.

    ``min_silhouette`` is a separate guard from the reproducibility guards and
    catches a different failure. Cutting a single unimodal cloud in half is highly
    *reproducible* — every seed finds the same arbitrary cut — while having almost
    no geometric separation. Reproducibility alone would bless that partition, so
    an absolute separation floor is required to say "this looks like a continuum,
    not groups".
    """
    usable = [b for b in benchmarks if b.silhouette is not None]
    if not usable:
        raise ValueError("No benchmark configuration produced a valid partition.")

    w = {
        "silhouette": 1.0,
        "calinski_harabasz": 0.5,
        "davies_bouldin": 0.5,
        "mean_bootstrap_jaccard": 1.5,
        "mean_ari_across_seeds": 1.0,
    }
    if weights:
        w.update(weights)

    columns = {
        "silhouette": _minmax([b.silhouette for b in usable]),
        "calinski_harabasz": _minmax([b.calinski_harabasz for b in usable]),
        "davies_bouldin": _minmax([b.davies_bouldin for b in usable], invert=True),
        "mean_bootstrap_jaccard": _minmax([b.mean_bootstrap_jaccard for b in usable]),
        "mean_ari_across_seeds": _minmax([b.mean_ari_across_seeds for b in usable]),
    }
    total_weight = sum(w[name] for name in columns)
    scores = [
        sum(w[name] * columns[name][i] for name in columns) / total_weight
        for i in range(len(usable))
    ]

    order = sorted(range(len(usable)), key=lambda i: scores[i], reverse=True)
    ranked = [
        (usable[i].representation, usable[i].algorithm, usable[i].k, float(scores[i]))
        for i in order
    ]
    best = usable[order[0]]

    warnings: list[str] = []
    if (best.silhouette or 0.0) < min_silhouette:
        warnings.append(
            f"best configuration silhouette {best.silhouette!r} < {min_silhouette}: the data "
            "may be a continuum rather than discrete groups, and the partition may be an "
            "arbitrary cut through it."
        )
    if (best.mean_bootstrap_jaccard or 0.0) < min_bootstrap_jaccard:
        warnings.append(
            f"best configuration bootstrap Jaccard "
            f"{best.mean_bootstrap_jaccard!r} < {min_bootstrap_jaccard}: "
            "the discovered groups are not stable enough to interpret individually."
        )
    if (best.mean_ari_across_seeds or 0.0) < min_ari:
        warnings.append(
            f"best configuration cross-seed ARI {best.mean_ari_across_seeds!r} < {min_ari}: "
            "the partition is seed-dependent and should be treated as exploratory only."
        )

    rationale = (
        f"K={best.k} selected for representation '{best.representation}' with "
        f"'{best.algorithm}' on measured evidence (silhouette={best.silhouette!r}, "
        f"bootstrap Jaccard={best.mean_bootstrap_jaccard!r}, "
        f"cross-seed ARI={best.mean_ari_across_seeds!r}) out of "
        f"{len(usable)} scored configurations spanning K in "
        f"{sorted({b.k for b in usable})}. No prior favoured any particular K."
    )
    return KSelection(
        k=best.k,
        representation=best.representation,
        algorithm=best.algorithm,
        score=float(scores[order[0]]),
        ranked=ranked,
        rationale=rationale,
        warnings=warnings,
    )
