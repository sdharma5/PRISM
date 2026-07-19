"""Continuous phenotype-profile similarity over normalized domain scores.

**These are not validated clinical subtypes.** No subtype label exists anywhere
in this repository and no external validation has been performed, so nothing here
may be described as diagnosing or assigning a PCOS subtype. The output is a
*similarity profile*: how much a patient's domain pattern resembles each of
several named research patterns from the literature. The language is enforced,
not merely recommended -- :mod:`models.phenotype.prototype_mapping` raises
``ProhibitedLanguageError`` on unhedged descriptions.

Why similarity in **domain space** rather than raw features: domain composites
are already coverage-aware and unit-free, they degrade gracefully when a variable
is missing, and a patient with no androgen assay simply carries no androgenic
coordinate rather than a silently-imputed zero. Matching on raw features would
make the profile depend on which assays a clinic happened to run.

Before any of that, a profile must be **eligible**: the domain that gives the
profile its name has to have been assessed for this patient. Similarity over
observed domains only is the right call, but on its own it lets a profile win on
its secondary weights while its defining axis was never measured -- which is how
patients came to be labelled ``androgenic_leaning`` in a cohort holding no
androgenic evidence. Ineligible profiles are *removed* from the catalogue for
that patient and the remaining similarities are renormalized over what is left;
they are never scored-then-suppressed and never zero-filled.

Routes to ``indeterminate``, all of which must exist or the profile overclaims:

1. **Thin evidence** -- too few domains observed to locate the patient at all.
2. **No eligible profile** -- nothing in the catalogue has its defining domain
   assessed here.
3. **Near-tie** -- the top two profiles are within a small margin, so the label
   is decoration rather than information.
4. **Weak match** -- the best similarity is low in absolute terms; the patient
   resembles nothing in the catalogue.
5. **Unstable** -- applied by :func:`summarize` from the stability engine's
   verdict: an assignment a single dropped domain would overturn is withheld.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.phenotype.indeterminate import INDETERMINATE, add_indeterminate_mass

__all__ = [
    "DOMAIN_PROTOTYPES",
    "MIXED_MIN_ASSESSABLE_DOMAINS",
    "PROFILE_DEFINING_DOMAINS",
    "PhenotypeSimilarity",
    "PrototypeSimilarityModel",
    "androgenic_evidence_source",
    "summarize",
]

#: Prototype centroids in DOMAIN space, in z-score units.
#:
#: Derived from the enrichment patterns in :mod:`prototype_rules`, which cite the
#: phenotype-subtyping literature (e.g. Dapas et al. 2020). They are declared
#: rather than fitted because 541 patients with no subtype labels cannot identify
#: centroids that would generalise; a fitted centroid here would be a cluster
#: artifact wearing a literature name. Fitting becomes appropriate only when
#: labelled or externally-validated subtypes exist.
DOMAIN_PROTOTYPES: dict[str, dict[str, float]] = {
    "metabolic_leaning": {
        "metabolic": 1.0,
        "symptom_burden": 0.3,
        "clinical_androgenic_evidence": 0.2,
        "biochemical_androgenic_evidence": 0.2,
        # Negative on the reproductive-axis pattern: the metabolic and LH/AMH
        # profiles are reported as contrasting, not merely different.
        "lh_amh_pattern": -0.5,
        "ovarian": -0.1,
    },
    "lh_amh_leaning": {
        "lh_amh_pattern": 1.0,
        "ovarian": 0.8,
        "reproductive": 0.6,
        "metabolic": -0.5,
        "symptom_burden": 0.1,
    },
    "androgenic_leaning": {
        # Clinical and biochemical androgenic evidence are separate domains and
        # either can define this profile, so both carry the defining weight.
        # Which one actually supported a given patient is recorded in
        # `androgenic_evidence_source`.
        "clinical_androgenic_evidence": 1.0,
        "biochemical_androgenic_evidence": 1.0,
        "symptom_burden": 0.7,
        "metabolic": 0.2,
        "reproductive": 0.3,
        "lh_amh_pattern": 0.2,
    },
    "mixed": {
        # Elevated across the board with no dominant axis. Deliberately modest
        # magnitudes so it wins only when nothing else does clearly.
        "reproductive": 0.5,
        "metabolic": 0.5,
        "clinical_androgenic_evidence": 0.5,
        "biochemical_androgenic_evidence": 0.5,
        "ovarian": 0.5,
        "lh_amh_pattern": 0.4,
        "symptom_burden": 0.5,
    },
}

#: Domains that DEFINE each profile. A profile is eligible only when at least one
#: of its defining domains was actually assessed.
#:
#: Without this a profile can win on its secondary weights while the axis that
#: gives it its name was never measured -- which is how patients were previously
#: labelled `androgenic_leaning` in a cohort carrying no androgenic evidence at
#: all. Similarity over observed domains only is still the right call; this rule
#: is what stops that choice from producing an unsupported label.
PROFILE_DEFINING_DOMAINS: dict[str, tuple[str, ...]] = {
    "metabolic_leaning": ("metabolic",),
    "lh_amh_leaning": ("lh_amh_pattern",),
    "androgenic_leaning": (
        "clinical_androgenic_evidence",
        "biochemical_androgenic_evidence",
    ),
}

#: `mixed` asserts elevation across several axes, so it needs at least this many
#: assessable domains before the claim means anything.
MIXED_MIN_ASSESSABLE_DOMAINS = 2

#: The two androgenic domains, for reporting which kind of evidence was available.
CLINICAL_ANDROGENIC = "clinical_androgenic_evidence"
BIOCHEMICAL_ANDROGENIC = "biochemical_androgenic_evidence"


def androgenic_evidence_source(domain_scores: dict[str, float | None]) -> str:
    """Which androgenic evidence a patient actually has.

    Returned verbatim on the output so that a downstream reader can never mistake
    cutaneous signs for a measured androgen level.
    """
    clinical = domain_scores.get(CLINICAL_ANDROGENIC) is not None
    biochemical = domain_scores.get(BIOCHEMICAL_ANDROGENIC) is not None
    if clinical and biochemical:
        return "both"
    if clinical:
        return "symptoms_only"
    if biochemical:
        return "biochemical_only"
    return "unavailable"


@dataclass
class PhenotypeSimilarity:
    """Soft profile similarity for one patient."""

    #: Normalized **affinity scores**, NOT calibrated probabilities.
    #:
    #: They sum to 1 and look like a posterior, but nothing here was calibrated
    #: against an outcome: there is no subtype ground truth to calibrate against,
    #: the centroids are declared rather than fitted, and the sharpness is set by
    #: an arbitrary softmax temperature. An affinity of 0.55 does NOT mean a 55%
    #: chance the patient "has" that profile. Read them as a ranking with a
    #: magnitude, and check `temperature_sensitivity` before trusting the gap.
    affinities: dict[str, float] = field(default_factory=dict)
    #: Raw cosine similarity per profile, before softmax or indeterminate mass.
    similarities: dict[str, float] = field(default_factory=dict)
    #: Euclidean distance to each centroid over the shared observed domains.
    distances: dict[str, float] = field(default_factory=dict)
    dominant: str | None = None
    entropy: float = 0.0
    observed_domains: list[str] = field(default_factory=list)
    missing_domains: list[str] = field(default_factory=list)
    indeterminate_reasons: list[str] = field(default_factory=list)
    supporting_evidence: dict[str, list[str]] = field(default_factory=dict)
    #: Profiles whose defining domain was assessed. Only these receive a
    #: similarity score, and affinities are renormalized over them alone --
    #: ineligible profiles are REMOVED, never zero-filled.
    eligible_profiles: list[str] = field(default_factory=list)
    #: Profile -> why it was excluded from scoring entirely.
    ineligible_profiles: dict[str, str] = field(default_factory=dict)
    #: One of: symptoms_only, biochemical_only, both, unavailable.
    androgenic_evidence_source: str = "unavailable"

    @property
    def is_indeterminate(self) -> bool:
        return self.dominant in (None, INDETERMINATE)


class PrototypeSimilarityModel:
    """Compare a patient's domain scores with named research prototypes."""

    version = "prototype-similarity-0.1.0"

    def __init__(
        self,
        prototypes: dict[str, dict[str, float]] | None = None,
        *,
        min_observed_domains: int = 3,
        near_tie_margin: float = 0.10,
        min_similarity: float = 0.30,
        temperature: float = 0.25,
    ) -> None:
        """
        Args:
            prototypes: Centroids in domain space.
            min_observed_domains: Below this the patient cannot be located.
            near_tie_margin: Top-two gap below which the assignment is a coin flip.
            min_similarity: Best similarity below which nothing matches well.
            temperature: Softmax temperature over similarities. Lower is sharper;
                0.25 keeps a 0.1 similarity gap from becoming a 0.9 probability.
        """
        self.prototypes = prototypes or DOMAIN_PROTOTYPES
        self.min_observed_domains = min_observed_domains
        self.near_tie_margin = near_tie_margin
        self.min_similarity = min_similarity
        self.temperature = temperature

    @staticmethod
    def _cosine(patient: dict[str, float], centroid: dict[str, float], keys: list[str]) -> float:
        a = np.array([patient.get(k, 0.0) for k in keys], dtype=float)
        b = np.array([centroid.get(k, 0.0) for k in keys], dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator > 1e-9 else 0.0

    def _eligible_profiles(
        self, observed: dict[str, float], result: PhenotypeSimilarity
    ) -> list[str]:
        """Profiles whose defining domain was actually assessed.

        Records the exclusion reason for every profile that is dropped, so the
        absence of a profile from the output is explainable rather than silent.
        """
        eligible: list[str] = []
        for name in self.prototypes:
            if name == "mixed":
                if len(observed) >= MIXED_MIN_ASSESSABLE_DOMAINS:
                    eligible.append(name)
                else:
                    result.ineligible_profiles[name] = (
                        f"`mixed` asserts elevation across axes and needs at least "
                        f"{MIXED_MIN_ASSESSABLE_DOMAINS} assessable domains; "
                        f"{len(observed)} were assessed"
                    )
                continue

            defining = PROFILE_DEFINING_DOMAINS.get(name)
            if defining is None:
                eligible.append(name)
                continue

            if any(domain in observed for domain in defining):
                eligible.append(name)
            else:
                result.ineligible_profiles[name] = (
                    f"defining domain(s) {', '.join(defining)} unavailable for this "
                    f"patient, so `{name}` cannot be supported by evidence"
                )

        result.eligible_profiles = eligible
        return eligible

    def predict(self, domain_scores: dict[str, float | None]) -> PhenotypeSimilarity:
        """Compute soft similarity to every prototype.

        Args:
            domain_scores: Domain name -> z-score composite, or None where the
                domain abstained for lack of coverage.

        Returns:
            A :class:`PhenotypeSimilarity`. Always carries an indeterminate
            entry, and the dominant profile is ``indeterminate`` whenever the
            evidence cannot support a named one.
        """
        observed = {
            name: float(value)
            for name, value in domain_scores.items()
            if value is not None and math.isfinite(float(value))
        }
        all_domains = sorted(set(domain_scores) | {k for p in self.prototypes.values() for k in p})
        missing = sorted(set(all_domains) - set(observed))

        result = PhenotypeSimilarity(
            observed_domains=sorted(observed),
            missing_domains=missing,
            androgenic_evidence_source=androgenic_evidence_source(domain_scores),
        )

        if len(observed) < self.min_observed_domains:
            result.indeterminate_reasons.append(
                f"only {len(observed)} domain score(s) available; at least "
                f"{self.min_observed_domains} are required to locate a phenotype profile"
            )
            result.affinities = {INDETERMINATE: 1.0}
            result.dominant = INDETERMINATE
            result.entropy = 0.0
            return result

        # Eligibility BEFORE scoring: a profile whose defining domain was never
        # assessed is removed from the catalogue for this patient rather than
        # scored and then suppressed. Scoring it first would leave a similarity
        # value on the output that a reader could quote.
        eligible = self._eligible_profiles(observed, result)
        if not eligible:
            result.indeterminate_reasons.append(
                "no profile's defining domain was assessed for this patient"
            )
            result.affinities = {INDETERMINATE: 1.0}
            result.dominant = INDETERMINATE
            result.entropy = 0.0
            return result

        # Compare only over domains the patient actually has. Zero-filling an
        # unobserved domain would let "not measured" read as "average", which for
        # a contrastive centroid (negative weights) actively distorts the match.
        keys = sorted(observed)
        result.similarities = {
            name: self._cosine(observed, self.prototypes[name], keys) for name in eligible
        }
        result.distances = {
            name: float(
                np.linalg.norm(
                    np.array([observed[k] for k in keys])
                    - np.array([self.prototypes[name].get(k, 0.0) for k in keys])
                )
            )
            for name in eligible
        }

        ordered = sorted(result.similarities.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_score = ordered[0]
        runner_up = ordered[1][1] if len(ordered) > 1 else -1.0

        # Softmax over similarities, then reserve indeterminate mass.
        values = np.array([s for _, s in result.similarities.items()], dtype=float)
        exponent = np.exp((values - values.max()) / max(self.temperature, 1e-6))
        soft = exponent / exponent.sum()
        affinities = dict(zip(result.similarities.keys(), (float(v) for v in soft), strict=True))

        indeterminate_mass = 0.0
        if best_score < self.min_similarity:
            indeterminate_mass = max(indeterminate_mass, 0.6)
            result.indeterminate_reasons.append(
                f"best similarity {best_score:.2f} is below the {self.min_similarity:.2f} "
                "floor: this patient resembles no catalogued profile closely"
            )
        if best_score - runner_up < self.near_tie_margin:
            indeterminate_mass = max(indeterminate_mass, 0.4)
            result.indeterminate_reasons.append(
                f"top two profiles are within {best_score - runner_up:.3f} "
                f"({best_name} vs {ordered[1][0]}): the assignment is close to a coin flip"
            )
        coverage = len(observed) / max(len(all_domains), 1)
        if coverage < 0.5:
            indeterminate_mass = max(indeterminate_mass, 1.0 - coverage)
            result.indeterminate_reasons.append(
                f"only {coverage:.0%} of phenotype domains observed"
            )

        result.affinities = add_indeterminate_mass(affinities, indeterminate_mass)

        top = max(result.affinities.items(), key=lambda kv: kv[1])
        result.dominant = top[0]
        result.entropy = float(
            -sum(p * math.log(max(p, 1e-12)) for p in result.affinities.values())
        )

        # Which domains actually drove each profile: the element-wise products
        # that contributed positively to the cosine.
        for name in eligible:
            centroid = self.prototypes[name]
            drivers = [
                domain for domain in keys if observed[domain] * centroid.get(domain, 0.0) > 0.15
            ]
            if drivers:
                result.supporting_evidence[name] = sorted(
                    drivers, key=lambda d: -abs(observed[d] * centroid.get(d, 0.0))
                )
        return result

    def temperature_sensitivity(
        self,
        domain_scores: dict[str, float | None],
        temperatures: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    ) -> dict[str, Any]:
        """How much the affinity ranking depends on the arbitrary temperature.

        The softmax temperature is a presentation choice, not a fitted parameter.
        If the dominant profile changes when it moves, the assignment was an
        artifact of that choice and must not be reported as a finding. This is
        the check that distinguishes "the evidence favours this profile" from
        "our sharpening constant favours this profile".
        """
        dominants: dict[str, str | None] = {}
        top_affinity: dict[str, float] = {}
        original = self.temperature
        try:
            for temperature in temperatures:
                self.temperature = temperature
                result = self.predict(domain_scores)
                dominants[str(temperature)] = result.dominant
                top_affinity[str(temperature)] = round(
                    max(result.affinities.values()) if result.affinities else 0.0, 4
                )
        finally:
            self.temperature = original

        distinct = {value for value in dominants.values()}
        return {
            "dominant_by_temperature": dominants,
            "top_affinity_by_temperature": top_affinity,
            "dominant_is_stable": len(distinct) == 1,
            "n_distinct_dominants": len(distinct),
        }

    def threshold_sensitivity(
        self,
        domain_scores: dict[str, float | None],
        margins: tuple[float, ...] = (0.05, 0.10, 0.20),
        similarity_floors: tuple[float, ...] = (0.20, 0.30, 0.40),
    ) -> dict[str, Any]:
        """How the indeterminate decision depends on its two declared cut-points.

        ``near_tie_margin`` and ``min_similarity`` are judgement calls. A patient
        whose determinacy flips across plausible settings is genuinely borderline,
        and reporting a confident profile for them would be overclaiming.
        """
        outcomes: dict[str, str | None] = {}
        original = (self.near_tie_margin, self.min_similarity)
        try:
            for margin in margins:
                for floor in similarity_floors:
                    self.near_tie_margin = margin
                    self.min_similarity = floor
                    outcomes[f"margin={margin},floor={floor}"] = self.predict(
                        domain_scores
                    ).dominant
        finally:
            self.near_tie_margin, self.min_similarity = original

        values = list(outcomes.values())
        indeterminate_fraction = sum(1 for v in values if v == INDETERMINATE) / max(len(values), 1)
        return {
            "dominant_by_threshold": outcomes,
            "n_distinct_dominants": len({v for v in values}),
            "indeterminate_fraction": round(indeterminate_fraction, 4),
            "decision_is_stable": len({v for v in values}) == 1,
        }

    def describe(self, similarity: PhenotypeSimilarity) -> str:
        """A hedged, human-readable statement of the result."""
        if similarity.is_indeterminate:
            reason = (
                similarity.indeterminate_reasons[0]
                if similarity.indeterminate_reasons
                else "evidence does not favour any profile"
            )
            return f"No phenotype profile is clearly indicated ({reason})."
        probability = similarity.affinities.get(similarity.dominant or "", 0.0)
        return (
            f"This patient's domain pattern most resembles the "
            f"'{similarity.dominant}' research profile (p={probability:.2f}). "
            "This is a similarity to a described research pattern, not a validated "
            "clinical subtype or a diagnosis."
        )


def summarize(similarity: PhenotypeSimilarity, *, is_stable: bool | None = None) -> dict[str, Any]:
    """Serializable summary for the token and report.

    Args:
        similarity: The prediction to serialize.
        is_stable: Verdict from :class:`~models.adapters.pcos.stability.
            PhenotypeStabilityEngine`. When it is ``False`` the dominant profile
            is withheld and the record is marked indeterminate: an assignment
            that a single dropped domain or a different softmax temperature
            would overturn is not a finding, and publishing it as one is the
            error this gate exists to prevent. ``None`` means stability was not
            assessed -- which also withholds the profile, because "we did not
            check" is not evidence of stability.
    """
    dominant = similarity.dominant
    reasons = list(similarity.indeterminate_reasons)

    if is_stable is not True and dominant not in (None, INDETERMINATE):
        reasons.append(
            f"assignment to '{dominant}' did not survive the stability checks "
            "(resampling, single-domain ablation, or temperature); it is reported "
            "as indeterminate rather than as a profile"
            if is_stable is False
            else f"assignment to '{dominant}' was never checked for stability, so it is "
            "reported as indeterminate rather than as a profile"
        )
        dominant = None

    indeterminate = dominant is None or dominant == INDETERMINATE
    return {
        "phenotype_affinities": {k: round(v, 4) for k, v in similarity.affinities.items()},
        "dominant_profile": None if indeterminate else dominant,
        "indeterminate": indeterminate,
        "indeterminate_reasons": reasons,
        "assignment_entropy": round(similarity.entropy, 4),
        "profile_similarities": {k: round(v, 4) for k, v in similarity.similarities.items()},
        "eligible_profiles": similarity.eligible_profiles,
        "ineligible_profiles": similarity.ineligible_profiles,
        "androgenic_evidence_source": similarity.androgenic_evidence_source,
        "observed_domains": similarity.observed_domains,
        "missing_domains": similarity.missing_domains,
        "stability_assessed": is_stable is not None,
        "assignment_is_stable": is_stable,
    }
