"""Describe discovered clusters by feature enrichment, in hedged language only.

Scientific WHY
--------------
A cluster index (``cluster_2``) is not communicable, but naming it is exactly
where unsupervised work goes wrong: a data-driven partition of one cohort is not
a validated clinical entity, and calling it one implies external validation,
prognostic meaning, and treatment relevance that no clustering run has earned.

So we do two things. First, descriptions are generated *mechanically* from
standardized enrichment versus the cohort mean — the description is a readout of
the data, not an interpretation layered on top. Second, every generated string
passes through :func:`assert_hedged_language`, which raises on a hard-coded list
of banned phrases. The guard is a unit-tested tripwire, not a style preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BANNED_PHRASES",
    "HEDGE_VERBS",
    "ClusterCharacterization",
    "ProhibitedLanguageError",
    "assert_hedged_language",
    "characterize_clusters",
    "describe_cluster",
]

#: Phrases that would assert clinical validity we do not have. Matched
#: case-insensitively as substrings against every generated description.
BANNED_PHRASES: tuple[str, ...] = (
    "clinically validated subtype",
    "validated subtype",
    "confirmed subtype",
    "diagnosis",
    "diagnosed",
    "diagnostic",
    "you have",
    "patient has pmos",
    "definitive",
    "definitively",
    "proven subtype",
    "established subtype",
    "clinical subtype",
    "disease subtype",
    "treatment should",
    "we recommend treatment",
    "rule out",
    "rules out",
)

#: The only verbs allowed to connect a participant or cluster to a profile name.
HEDGE_VERBS: tuple[str, ...] = (
    "resembles",
    "is most similar to",
    "has overlap with",
    "shows a pattern consistent with",
)


class ProhibitedLanguageError(ValueError):
    """Raised when generated text asserts clinical validity we cannot support."""


def assert_hedged_language(text: str, context: str = "description") -> str:
    """Raise :class:`ProhibitedLanguageError` if ``text`` contains a banned phrase.

    Called on every description this module emits and re-callable by adapters on
    any user-visible string. Returns ``text`` unchanged so it can wrap a return.
    """
    lowered = text.lower()
    hits = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
    if hits:
        raise ProhibitedLanguageError(
            f"{context} contains prohibited non-hedged phrasing {hits!r}. Discovered "
            f"groups are exploratory data-driven profiles, never validated clinical "
            f"subtypes. Offending text: {text!r}"
        )
    return text


@dataclass(frozen=True)
class ClusterCharacterization:
    """Standardized enrichment of one discovered cluster versus the cohort."""

    cluster: str
    n_members: int
    enrichment: dict[str, float]
    elevated: list[str] = field(default_factory=list)
    reduced: list[str] = field(default_factory=list)
    description: str = ""


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z-scored matrix, cohort mean, cohort sd) with zero-variance guard."""
    mean = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mean) / sd, mean, sd


def describe_cluster(
    cluster: str,
    enrichment: dict[str, float],
    n_members: int,
    threshold: float = 0.4,
    max_terms: int = 4,
) -> str:
    """Build one hedged, mechanical description from an enrichment vector.

    ``threshold`` is in cohort standard deviations. We report only the strongest
    ``max_terms`` deviations in each direction: a description listing every
    feature is unreadable and invites over-reading small differences.
    """
    ordered = sorted(enrichment.items(), key=lambda kv: abs(kv[1]), reverse=True)
    up = [name for name, value in ordered if value >= threshold][:max_terms]
    down = [name for name, value in ordered if value <= -threshold][:max_terms]

    parts: list[str] = []
    if up:
        parts.append("higher than the cohort average for " + ", ".join(up))
    if down:
        parts.append("lower than the cohort average for " + ", ".join(down))
    body = (
        "; ".join(parts)
        if parts
        else (
            "no feature deviating from the cohort average by more than "
            f"{threshold} standard deviations"
        )
    )
    text = (
        f"Group '{cluster}' (n={n_members}) is a data-driven grouping found in this "
        f"cohort only. On average its members show {body}. This is an exploratory "
        f"pattern, not a validated clinical category."
    )
    return assert_hedged_language(text, context=f"description for {cluster}")


def characterize_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.4,
    label_prefix: str = "profile_",
) -> dict[str, ClusterCharacterization]:
    """Characterize every discovered cluster by mean z-score versus the cohort.

    Enrichment is the cluster's mean of cohort-standardized features, i.e. how
    many cohort standard deviations the cluster centre sits from the cohort
    centre. This is directly comparable across features with different units,
    which is why we standardize rather than report raw means.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.shape[1] != len(feature_names):
        raise ValueError(f"{X.shape[1]} columns but {len(feature_names)} feature names supplied.")
    Z, _, _ = _standardize(X)

    out: dict[str, ClusterCharacterization] = {}
    for raw_label in np.unique(labels):
        mask = labels == raw_label
        name = f"{label_prefix}{raw_label}"
        means = np.nanmean(Z[mask], axis=0)
        enrichment = {feature_names[i]: float(means[i]) for i in range(len(feature_names))}
        out[name] = ClusterCharacterization(
            cluster=name,
            n_members=int(mask.sum()),
            enrichment=enrichment,
            elevated=sorted(
                (f for f, v in enrichment.items() if v >= threshold),
                key=lambda f: -enrichment[f],
            ),
            reduced=sorted(
                (f for f, v in enrichment.items() if v <= -threshold),
                key=lambda f: enrichment[f],
            ),
            description=describe_cluster(name, enrichment, int(mask.sum()), threshold),
        )
    return out
