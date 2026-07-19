"""Per-patient stability of the phenotype-profile assignment.

An affinity score says which profile a patient most resembles. It says nothing
about whether that answer would survive a small perturbation — and on six domain
scores, several of which may be absent, the answer often does not. This module
answers the second question, and its output is what licenses (or withholds) the
profile statement in the report.

Four perturbations, each probing a different fragility:

* **Domain ablation** — drop one domain at a time. If removing a single score
  changes the dominant profile, that profile was one measurement deep.
* **Noise resampling** — jitter the scores by their plausible measurement error
  and re-assign. The fraction of resamples agreeing with the point estimate is
  the bootstrap agreement.
* **Modality removal** — drop the domains a whole modality contributed. This is
  the deployment-realistic case: a patient without labs is not a patient with
  noisy labs.
* **Threshold / temperature** — delegated to
  :class:`~models.adapters.pcos.prototype_similarity.PrototypeSimilarityModel`,
  since those cut-points are declared rather than measured.

An unstable assignment is not an error to be smoothed away. It is the correct
finding for a patient who sits between profiles, and the report must say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.phenotype.indeterminate import INDETERMINATE

__all__ = ["PhenotypeStabilityReport", "PhenotypeStabilityEngine"]

#: Domains each modality can supply, for modality-removal sensitivity.
_MODALITY_DOMAINS: dict[str, tuple[str, ...]] = {
    "static_clinical": (
        "reproductive",
        "metabolic",
        "clinical_androgenic_evidence",
        "biochemical_androgenic_evidence",
        "ovarian",
        "lh_amh_pattern",
        "symptom_burden",
    ),
    "ovarian_ultrasound": ("ovarian",),
    "longitudinal_hormonal_state": ("reproductive",),
}


@dataclass
class PhenotypeStabilityReport:
    """Everything known about how fragile one patient's assignment is."""

    dominant_profile: str | None = None
    affinities: dict[str, float] = field(default_factory=dict)
    assignment_entropy: float = 0.0

    bootstrap_agreement: float = 0.0
    n_bootstrap: int = 0
    profile_flip_rate: float = 0.0

    domain_ablation: dict[str, str | None] = field(default_factory=dict)
    unstable_domains: list[str] = field(default_factory=list)
    modality_removal: dict[str, str | None] = field(default_factory=dict)
    unstable_modalities: list[str] = field(default_factory=list)

    temperature_sensitivity: dict[str, Any] = field(default_factory=dict)
    threshold_sensitivity: dict[str, Any] = field(default_factory=dict)

    n_observed_domains: int = 0
    meets_minimum_domains: bool = False
    stability_score: float = 0.0
    is_stable: bool = False
    abstain_from_profile: bool = False
    abstention_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    # Kept for the adapter's existing field name.
    @property
    def subtype_flip_rate(self) -> float:
        return self.profile_flip_rate


class PhenotypeStabilityEngine:
    """Perturb a patient's domain scores and see whether the profile survives."""

    version = "phenotype-stability-0.1.0"

    def __init__(
        self,
        *,
        n_bootstrap: int = 200,
        noise_scale: float = 0.25,
        seed: int = 0,
        min_observed_domains: int = 3,
        min_bootstrap_agreement: float = 0.60,
        max_unstable_domains: int = 1,
    ) -> None:
        """
        Args:
            n_bootstrap: Noise resamples.
            noise_scale: Jitter standard deviation in z-score units. 0.25 reflects
                that a domain composite is a weighted mean of several noisy
                measurements, so it is more stable than any single one.
            seed: Deterministic resampling.
            min_observed_domains: Hard floor; below this no profile is reported.
            min_bootstrap_agreement: Agreement below which the profile is unstable.
            max_unstable_domains: How many single-domain ablations may flip the
                assignment before it is called fragile.
        """
        self.n_bootstrap = n_bootstrap
        self.noise_scale = noise_scale
        self.seed = seed
        self.min_observed_domains = min_observed_domains
        self.min_bootstrap_agreement = min_bootstrap_agreement
        self.max_unstable_domains = max_unstable_domains

    def evaluate(
        self,
        domain_scores: dict[str, float | None],
        model: Any,
        *,
        available_modalities: list[str] | None = None,
    ) -> PhenotypeStabilityReport:
        """Assess assignment stability for one patient.

        Args:
            domain_scores: Domain -> z-score or None.
            model: A ``PrototypeSimilarityModel``.
            available_modalities: Modalities present, for removal sensitivity.

        Returns:
            A populated :class:`PhenotypeStabilityReport`.
        """
        observed = {
            name: float(value)
            for name, value in domain_scores.items()
            if value is not None and np.isfinite(float(value))
        }
        report = PhenotypeStabilityReport(n_observed_domains=len(observed))
        report.meets_minimum_domains = len(observed) >= self.min_observed_domains

        base = model.predict(domain_scores)
        report.dominant_profile = base.dominant
        report.affinities = dict(base.affinities)
        report.assignment_entropy = base.entropy

        if not report.meets_minimum_domains:
            # The explicit minimum-observed-domain rule. Below it there is nothing
            # to be stable *about*, and a bootstrap over 2 domains would report a
            # confident agreement for an assignment that means nothing.
            report.abstain_from_profile = True
            report.abstention_reason = (
                f"Only {len(observed)} domain score(s) observed; at least "
                f"{self.min_observed_domains} are required before a phenotype profile "
                "may be reported."
            )
            report.warnings.append(report.abstention_reason)
            return report

        # -- domain ablation ------------------------------------------------
        for domain in sorted(observed):
            ablated = {k: v for k, v in domain_scores.items() if k != domain}
            outcome = model.predict(ablated).dominant
            report.domain_ablation[domain] = outcome
            if outcome != base.dominant:
                report.unstable_domains.append(domain)

        # -- modality removal ------------------------------------------------
        for modality in available_modalities or []:
            contributed = _MODALITY_DOMAINS.get(modality, ())
            reduced = {k: v for k, v in domain_scores.items() if k not in contributed}
            outcome = model.predict(reduced).dominant
            report.modality_removal[modality] = outcome
            if outcome != base.dominant:
                report.unstable_modalities.append(modality)

        # -- noise resampling ------------------------------------------------
        rng = np.random.default_rng(self.seed)
        agreements = 0
        for _ in range(self.n_bootstrap):
            jittered = {
                name: value + float(rng.normal(0.0, self.noise_scale))
                for name, value in observed.items()
            }
            if model.predict(jittered).dominant == base.dominant:
                agreements += 1
        report.n_bootstrap = self.n_bootstrap
        report.bootstrap_agreement = agreements / max(self.n_bootstrap, 1)
        report.profile_flip_rate = 1.0 - report.bootstrap_agreement

        # -- declared cut-points ---------------------------------------------
        report.temperature_sensitivity = model.temperature_sensitivity(domain_scores)
        report.threshold_sensitivity = model.threshold_sensitivity(domain_scores)

        # -- verdict -----------------------------------------------------------
        # Product rather than mean: a profile that survives noise but flips when
        # one domain is dropped is not two-thirds stable, it is fragile. Any one
        # factor collapsing should collapse the score.
        ablation_factor = 1.0 - (len(report.unstable_domains) / max(len(observed), 1))
        temperature_factor = 1.0 if report.temperature_sensitivity["dominant_is_stable"] else 0.5
        threshold_factor = 1.0 if report.threshold_sensitivity["decision_is_stable"] else 0.6
        combined = (
            report.bootstrap_agreement * ablation_factor * temperature_factor * threshold_factor
        )
        report.stability_score = float(np.clip(combined, 0.0, 1.0))

        report.is_stable = (
            report.bootstrap_agreement >= self.min_bootstrap_agreement
            and len(report.unstable_domains) <= self.max_unstable_domains
            and report.temperature_sensitivity["dominant_is_stable"]
        )

        if base.dominant == INDETERMINATE:
            report.warnings.append(
                "Assignment is indeterminate; stability statistics describe the "
                "stability of that indeterminacy, not of a named profile."
            )
        if not report.is_stable:
            report.warnings.append(
                f"Profile assignment is unstable (bootstrap agreement "
                f"{report.bootstrap_agreement:.2f}, {len(report.unstable_domains)} "
                "domain(s) flip it). Report it as provisional or not at all."
            )
        if report.unstable_domains:
            report.warnings.append(
                "Removing any of these domains changes the assignment: "
                + ", ".join(report.unstable_domains)
            )
        return report
