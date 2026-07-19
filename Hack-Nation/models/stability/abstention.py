"""The indeterminate / abstain decision for one participant.

Scientific WHY
--------------
Forcing a label onto every participant is the single most harmful thing an
unsupervised phenotyping tool can do, because the participants hardest to place
are exactly the ones whose report will be over-read. Abstention is therefore a
first-class output, not an error path.

Six independent conditions can each trigger abstention. They are independent on
purpose: each catches a different way the assignment can be untrustworthy, and a
participant only has to fail one to be reported as indeterminate.

1. **Low confidence** — the top calibrated membership probability is below
   threshold. The participant sits between profiles.
2. **Model disagreement** — different representations or algorithms place the
   participant in different groups. The label is an artifact of an analyst choice.
3. **Unstable bootstrap assignment** — the participant (or the cluster they were
   put in) does not survive resampling.
4. **Single-variable fragility** — removing one variable flips the dominant
   group. The assignment is hostage to one measurement.
5. **Far from every centre** — a large scaled/Mahalanobis distance to the nearest
   centre means the participant is an outlier that the partition merely absorbed;
   K-means in particular assigns such a point to *something* regardless.
6. **Too little evidence** — too few of the variables that define the profiles
   were actually observed. A confident label from three observed variables is
   confidence about imputation, not about the participant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from evaluation.stability import assignment_entropy
from schemas.phenotype import INDETERMINATE, StabilityReport

__all__ = [
    "REASON_CODES",
    "AbstentionEvidence",
    "AbstentionThresholds",
    "evaluate_abstention",
    "scaled_distance_percentile",
]

REASON_CODES: tuple[str, ...] = (
    "low_confidence",
    "model_disagreement",
    "unstable_bootstrap_assignment",
    "single_variable_fragility",
    "far_from_all_profiles",
    "insufficient_observed_evidence",
)


@dataclass(frozen=True)
class AbstentionThresholds:
    """Documented, configurable decision boundaries.

    Defaults are deliberately conservative for a research artifact: we would
    rather return "indeterminate" for a participant who could have been placed
    than place one who should not have been.
    """

    #: Rule 1. Below this top-1 calibrated probability, no profile is named.
    min_dominant_probability: float = 0.50
    #: Rule 2. Fraction of alternative (representation, algorithm) fits that must
    #: agree with the dominant assignment.
    min_model_agreement: float = 0.60
    #: Rule 3a. Per-participant bootstrap agreement rate.
    min_bootstrap_agreement: float = 0.60
    #: Rule 3b. Hennig's per-cluster Jaccard for the assigned cluster.
    min_cluster_jaccard: float = 0.60
    #: Rule 4. Any single feature whose removal flips this participant abstains
    #: when True; otherwise the fraction of flipping features must stay below
    #: ``max_fragile_feature_fraction``.
    abstain_on_any_single_feature_flip: bool = True
    max_fragile_feature_fraction: float = 0.10
    #: Rule 5. Percentile of the cohort's nearest-centre distance distribution
    #: above which a participant counts as far from every profile.
    max_distance_percentile: float = 0.95
    #: Rule 6. Fraction of the profile-defining variables that must be observed.
    min_defining_feature_coverage: float = 0.50
    #: Reported entropy above this adds a warning (not on its own an abstention).
    warn_assignment_entropy: float = 0.80


@dataclass
class AbstentionEvidence:
    """Everything the six rules need, gathered for one participant.

    Any field left ``None`` disables its rule and records a warning, so a caller
    that has not run (say) the ablation sweep gets an honest "this check was not
    performed" rather than a silent pass.
    """

    patient_id: str
    dominant_profile: str
    probabilities: Mapping[str, float]
    #: Rule 2: dominant label from each alternative (representation, algorithm) fit.
    alternative_assignments: Sequence[str] | None = None
    #: Rule 3a / 3b.
    bootstrap_agreement: float | None = None
    cluster_jaccard: float | None = None
    #: Rule 4: names of features whose removal flips this participant.
    flipping_features: Sequence[str] | None = None
    n_features_tested: int | None = None
    fragility_by_feature: Mapping[str, float] | None = None
    #: Rule 5.
    distance_percentile: float | None = None
    #: Rule 6.
    n_defining_features_observed: int | None = None
    n_defining_features_total: int | None = None
    #: Carried through to the report.
    subtype_flip_rate: float = 0.0
    extra_warnings: list[str] = field(default_factory=list)


def scaled_distance_percentile(
    x: np.ndarray,
    centers: np.ndarray,
    cohort: np.ndarray,
    covariance: np.ndarray | None = None,
) -> float:
    """Where this participant's nearest-centre distance falls in the cohort.

    Uses Mahalanobis distance under the pooled covariance when that covariance is
    invertible, which correctly discounts distance along directions the cohort
    naturally varies in; falls back to Euclidean distance in the (already
    standardized) representation otherwise. Returning a *percentile* rather than a
    raw distance makes the threshold interpretable and unit-free.
    """
    x = np.asarray(x, dtype=float).ravel()
    centers = np.asarray(centers, dtype=float)
    cohort = np.asarray(cohort, dtype=float)

    inv: np.ndarray | None = None
    cov = np.cov(cohort, rowvar=False) if covariance is None else np.asarray(covariance)
    cov = np.atleast_2d(cov)
    if cov.shape[0] == cov.shape[1] == x.size:
        try:
            inv = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-8)
        except np.linalg.LinAlgError:
            inv = None

    def _distance(point: np.ndarray) -> float:
        deltas = centers - point[None, :]
        if inv is None:
            return float(np.sqrt((deltas**2).sum(axis=1)).min())
        return float(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", deltas, inv, deltas), 0.0)).min())

    own = _distance(x)
    cohort_distances = np.array([_distance(row) for row in cohort])
    if cohort_distances.size == 0:
        return 0.0
    return float((cohort_distances <= own).mean())


def evaluate_abstention(
    evidence: AbstentionEvidence,
    thresholds: AbstentionThresholds | None = None,
) -> StabilityReport:
    """Apply every abstention rule and build the participant's StabilityReport.

    The returned report always names the *candidate* dominant profile, even when
    abstaining, so a reader can see what was rejected and why. Consumers must gate
    on ``abstain``, and the adapter additionally moves the probability mass to
    ``indeterminate``.
    """
    t = thresholds or AbstentionThresholds()
    reasons: list[str] = []
    warnings: list[str] = list(evidence.extra_warnings)

    probabilities = {str(k): float(v) for k, v in evidence.probabilities.items()}
    dominant_probability = float(probabilities.get(evidence.dominant_profile, 0.0))
    entropy = assignment_entropy([v for k, v in probabilities.items() if k != INDETERMINATE])

    # Rule 1 — low confidence.
    if dominant_probability < t.min_dominant_probability:
        reasons.append(
            f"low_confidence: top membership probability {dominant_probability:.2f} < "
            f"{t.min_dominant_probability:.2f}"
        )

    # Rule 2 — disagreement between representations / algorithms.
    if evidence.alternative_assignments is None:
        warnings.append("model_disagreement check not performed: no alternative fits supplied")
    elif len(evidence.alternative_assignments) > 0:
        agreement = sum(
            1 for a in evidence.alternative_assignments if a == evidence.dominant_profile
        ) / len(evidence.alternative_assignments)
        if agreement < t.min_model_agreement:
            reasons.append(
                f"model_disagreement: only {agreement:.0%} of "
                f"{len(evidence.alternative_assignments)} alternative fits agree "
                f"(< {t.min_model_agreement:.0%})"
            )

    # Rule 3 — unstable under bootstrap resampling.
    if evidence.bootstrap_agreement is None and evidence.cluster_jaccard is None:
        warnings.append("bootstrap stability check not performed: no bootstrap evidence supplied")
    if (
        evidence.bootstrap_agreement is not None
        and evidence.bootstrap_agreement < t.min_bootstrap_agreement
    ):
        reasons.append(
            f"unstable_bootstrap_assignment: this participant kept the same group in only "
            f"{evidence.bootstrap_agreement:.0%} of resamples "
            f"(< {t.min_bootstrap_agreement:.0%})"
        )
    if evidence.cluster_jaccard is not None and evidence.cluster_jaccard < t.min_cluster_jaccard:
        reasons.append(
            f"unstable_bootstrap_assignment: the assigned group's bootstrap Jaccard "
            f"{evidence.cluster_jaccard:.2f} < {t.min_cluster_jaccard:.2f}, so the group "
            f"itself does not reliably reappear"
        )

    # Rule 4 — one variable decides the answer.
    if evidence.flipping_features is None:
        warnings.append("single_variable_fragility check not performed: no ablation supplied")
    else:
        flipping = list(evidence.flipping_features)
        tested = evidence.n_features_tested or 0
        if flipping and t.abstain_on_any_single_feature_flip:
            reasons.append(
                "single_variable_fragility: removing "
                + ", ".join(sorted(flipping)[:3])
                + (" (and others)" if len(flipping) > 3 else "")
                + " alone changes the dominant group"
            )
        elif flipping and tested > 0:
            fraction = len(flipping) / tested
            if fraction > t.max_fragile_feature_fraction:
                reasons.append(
                    f"single_variable_fragility: {fraction:.0%} of single-variable removals "
                    f"change the dominant group (> {t.max_fragile_feature_fraction:.0%})"
                )

    # Rule 5 — outlier with respect to every profile centre.
    if evidence.distance_percentile is None:
        warnings.append("distance_to_profiles check not performed: no distance percentile supplied")
    elif evidence.distance_percentile > t.max_distance_percentile:
        reasons.append(
            f"far_from_all_profiles: distance to the nearest profile centre is at the "
            f"{evidence.distance_percentile:.0%} percentile of the cohort "
            f"(> {t.max_distance_percentile:.0%}); the participant resembles none of them"
        )

    # Rule 6 — not enough of the defining variables were measured.
    total = evidence.n_defining_features_total
    observed = evidence.n_defining_features_observed
    if total is None or observed is None:
        warnings.append("evidence_coverage check not performed: defining-feature counts missing")
    elif total > 0:
        coverage = observed / total
        if coverage < t.min_defining_feature_coverage:
            reasons.append(
                f"insufficient_observed_evidence: only {observed}/{total} "
                f"({coverage:.0%}) of the profile-defining variables were observed "
                f"(< {t.min_defining_feature_coverage:.0%})"
            )

    if entropy > t.warn_assignment_entropy:
        warnings.append(
            f"membership entropy {entropy:.2f} is high: the probabilities are close to uniform"
        )

    fragility = dict(evidence.fragility_by_feature or {})
    highest = max(fragility.items(), key=lambda kv: kv[1])[0] if fragility else None

    stability_components = [
        v
        for v in (
            evidence.bootstrap_agreement,
            evidence.cluster_jaccard,
            1.0 - float(evidence.subtype_flip_rate),
        )
        if v is not None
    ]
    stability_score = float(np.clip(np.mean(stability_components), 0.0, 1.0))

    return StabilityReport(
        patient_id=evidence.patient_id,
        dominant_profile=evidence.dominant_profile,
        dominant_probability=dominant_probability,
        stability_score=stability_score,
        subtype_flip_rate=float(np.clip(evidence.subtype_flip_rate, 0.0, 1.0)),
        assignment_entropy=entropy,
        bootstrap_jaccard=evidence.cluster_jaccard,
        highest_fragility_feature=highest,
        fragility_by_feature=fragility,
        abstain=bool(reasons),
        abstain_reasons=reasons,
        warnings=warnings,
    )
