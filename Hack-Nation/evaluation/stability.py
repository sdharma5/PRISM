"""Stability statistics: does a discovered grouping survive being disturbed?

Scientific WHY
--------------
A clustering solution is a *hypothesis*, and the cheapest falsification test is
perturbation. We implement four families of statistic:

1. **Bootstrap Jaccard** (Hennig 2007, "Cluster-wise assessment of cluster
   stability"): resample participants, re-cluster, and match each original
   cluster to its most similar bootstrap cluster by Jaccard overlap. The
   conventional reading is that mean Jaccard < 0.60 indicates a cluster that
   should not be interpreted, 0.60-0.75 indicates a pattern worth reporting only
   with caveats, and > 0.85 indicates a highly stable pattern.
2. **Assignment entropy**: how undecided a *patient's* soft membership is.
3. **Subtype flip rate**: how often a patient's dominant label changes under
   perturbation. This is the number a reader actually cares about.
4. **Jensen-Shannon divergence**: a bounded, symmetric distance between the
   original and perturbed membership distributions, so "the label survived but
   the confidence collapsed" is still visible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "assignment_entropy",
    "bootstrap_jaccard",
    "jaccard",
    "jensen_shannon_divergence",
    "match_clusters",
    "subtype_flip_rate",
]

_EPS = 1e-12


def jaccard(a: set[int] | set[str], b: set[int] | set[str]) -> float:
    """Jaccard similarity |a ∩ b| / |a ∪ b|; 0.0 when both are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def match_clusters(
    reference: np.ndarray,
    candidate: np.ndarray,
    index: Sequence[int] | None = None,
) -> dict[int, float]:
    """Best-matching Jaccard for each reference cluster against ``candidate``.

    ``index`` gives the positions in the reference labelling that ``candidate``
    covers, which is what makes this usable for bootstrap samples that only
    contain a subset (or multiset) of the original participants.

    The reference cluster is **restricted to the covered positions** before the
    overlap is taken. Without that restriction a bootstrap draw with replacement
    would cap Jaccard at roughly 0.63 (the expected unique fraction of an n-of-n
    resample) even for a perfectly recovered partition, and every cluster would
    look unstable for a purely combinatorial reason.
    """
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    positions = np.asarray(index) if index is not None else np.arange(len(candidate))
    covered = {int(i) for i in positions}

    out: dict[int, float] = {}
    for ref_label in np.unique(reference):
        ref_members = {int(i) for i in np.flatnonzero(reference == ref_label)} & covered
        best = 0.0
        for cand_label in np.unique(candidate):
            cand_members = {int(positions[i]) for i in np.flatnonzero(candidate == cand_label)}
            best = max(best, jaccard(ref_members, cand_members))
        out[int(ref_label)] = best
    return out


def bootstrap_jaccard(
    reference: np.ndarray,
    bootstrap_labelings: Sequence[tuple[np.ndarray, Sequence[int]]],
) -> dict[str, float]:
    """Per-cluster and mean bootstrap Jaccard stability.

    ``bootstrap_labelings`` is a sequence of ``(labels, original_indices)``
    pairs, one per resample.
    """
    reference = np.asarray(reference)
    per_cluster: dict[int, list[float]] = {int(k): [] for k in np.unique(reference)}
    for labels, idx in bootstrap_labelings:
        matched = match_clusters(reference, labels, idx)
        for cluster, score in matched.items():
            per_cluster.setdefault(cluster, []).append(score)

    result = {
        f"cluster_{cluster}": float(np.mean(scores)) if scores else 0.0
        for cluster, scores in sorted(per_cluster.items())
    }
    result["mean"] = float(np.mean(list(result.values()))) if result else 0.0
    return result


def assignment_entropy(probabilities: Sequence[float], normalize: bool = True) -> float:
    """Shannon entropy of a membership vector, optionally scaled to [0, 1].

    Normalizing by ``log(k)`` makes entropies comparable across different K,
    which matters because our benchmark sweeps K from 2 to 6.
    """
    p = np.asarray(list(probabilities), dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    entropy = float(-(p * np.log(p)).sum())
    if normalize and p.size > 1:
        entropy /= float(np.log(p.size))
    return entropy


def subtype_flip_rate(original: Sequence[str | int], perturbed: Sequence[str | int]) -> float:
    """Fraction of participants whose dominant label changed under perturbation."""
    if len(original) != len(perturbed):
        raise ValueError("original and perturbed labellings must have equal length.")
    if not original:
        return 0.0
    flips = sum(1 for a, b in zip(original, perturbed, strict=True) if a != b)
    return flips / len(original)


def jensen_shannon_divergence(
    p: Sequence[float] | dict[str, float],
    q: Sequence[float] | dict[str, float],
    base: float = 2.0,
) -> float:
    """Jensen-Shannon divergence between two membership distributions.

    Bounded in [0, 1] with ``base=2``, symmetric, and finite even when one
    distribution puts zero mass where the other does not — all three properties
    are why we use it instead of KL divergence for reporting drift.
    """
    if isinstance(p, dict) or isinstance(q, dict):
        pd = dict(p) if isinstance(p, dict) else {}
        qd = dict(q) if isinstance(q, dict) else {}
        keys = sorted(set(pd) | set(qd))
        pv = np.array([pd.get(k, 0.0) for k in keys], dtype=float)
        qv = np.array([qd.get(k, 0.0) for k in keys], dtype=float)
    else:
        pv = np.asarray(list(p), dtype=float)
        qv = np.asarray(list(q), dtype=float)
        if pv.shape != qv.shape:
            raise ValueError("probability vectors must have the same length.")

    pv = pv / pv.sum() if pv.sum() > 0 else np.full_like(pv, 1.0 / max(pv.size, 1))
    qv = qv / qv.sum() if qv.sum() > 0 else np.full_like(qv, 1.0 / max(qv.size, 1))
    m = 0.5 * (pv + qv)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log((a[mask] + _EPS) / (b[mask] + _EPS))).sum())

    div = 0.5 * _kl(pv, m) + 0.5 * _kl(qv, m)
    div /= float(np.log(base))
    return float(max(0.0, min(1.0, div)))
