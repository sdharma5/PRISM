"""Turn domain scores plus cluster membership into a :class:`PhenotypeProfile`.

Scientific WHY
--------------
Two different things are being combined here and they must not be confused.

*Domain scores* are transparent, weighted composites over registry-declared
variables. They are interpretable and they carry coverage: a metabolic score
computed from two of eleven variables is a different object from one computed
from ten, and the schema forces us to say which we have.

*Cluster membership* is the output of unsupervised discovery — probabilistic,
unstable, and only meaningful relative to the cohort it was discovered in.

The profile keeps both, plus the evidence qualifiers that stop a symptom-only
androgenic score being read as biochemical hyperandrogenism. When a domain's
coverage falls under its registry-declared ``min_coverage_to_report`` the score is
withheld (set to ``None``) rather than reported with a caveat, because a number
on a page outlives its footnote.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from models.phenotype.indeterminate import add_indeterminate_mass, dominant_with_indeterminate
from registry.loader import load_phenotype_domains
from schemas.phenotype import DomainScore, PhenotypeProfile

__all__ = [
    "compute_domain_scores",
    "defining_feature_coverage",
    "build_phenotype_profile",
]


def _domain_specs() -> dict[str, Any]:
    return dict(load_phenotype_domains().get("domains", {}))


def compute_domain_scores(
    standardized_values: Mapping[str, float | None],
    domains: Mapping[str, Any] | None = None,
) -> dict[str, DomainScore]:
    """Weighted mean of observed z-scores per domain, with coverage accounting.

    ``standardized_values`` must already be cohort-standardized (z-scores), so
    that variables in different units are combined on a common scale. Coverage is
    *weight-weighted*, not a raw count: missing the single highest-weighted
    variable in a domain should cost more than missing a minor one.
    """
    specs = dict(domains) if domains is not None else _domain_specs()
    out: dict[str, DomainScore] = {}

    for name, spec in specs.items():
        features = list(spec.get("features", []))
        min_coverage = float(spec.get("min_coverage_to_report", 0.0))
        qualifier_text = spec.get("symptom_only_qualifier")

        total_weight = sum(float(f.get("weight", 1.0)) for f in features) or 1.0
        observed: list[str] = []
        missing: list[str] = []
        numerator = 0.0
        observed_weight = 0.0
        observed_classes: set[str] = set()

        for feature in features:
            code = str(feature["code"])
            value = standardized_values.get(code)
            if value is None:
                missing.append(code)
                continue
            weight = float(feature.get("weight", 1.0))
            direction = float(feature.get("direction", 1))
            numerator += weight * direction * float(value)
            observed_weight += weight
            observed.append(code)
            observed_classes.add(str(feature.get("evidence_class", "unknown")))

        coverage = observed_weight / total_weight
        warnings: list[str] = []
        score: float | None = None

        if observed_weight <= 0:
            warnings.append(f"domain '{name}' has no observed variables; no score is reported")
        elif coverage < min_coverage:
            warnings.append(
                f"domain '{name}' coverage {coverage:.0%} is below the registry minimum "
                f"{min_coverage:.0%}; the score is withheld rather than reported weakly"
            )
        else:
            score = numerator / observed_weight

        qualifier: str | None = None
        if qualifier_text and observed_classes and observed_classes <= {"report"}:
            qualifier = str(qualifier_text)
            warnings.append(
                f"domain '{name}' is supported only by reported symptoms; it is "
                f"qualified as '{qualifier}' and is not biochemical evidence"
            )

        out[name] = DomainScore(
            domain=name,
            score=score,
            coverage=float(min(max(coverage, 0.0), 1.0)),
            observed_features=observed,
            missing_features=missing,
            evidence_qualifier=qualifier,
            evidence_source=str(spec.get("evidence_source", "mixed")),
            warnings=warnings,
        )
    return out


def defining_feature_coverage(
    standardized_values: Mapping[str, float | None],
    defining_features: list[str],
) -> tuple[int, int]:
    """(observed, total) count over the variables that define the discovered profiles.

    Feeds abstention rule 6: a confident membership computed from mostly imputed
    inputs is confidence about the imputer, not about the participant.
    """
    total = len(defining_features)
    observed = sum(1 for code in defining_features if standardized_values.get(code) is not None)
    return observed, total


def build_phenotype_profile(
    patient_id: str,
    domain_scores: Mapping[str, DomainScore],
    membership: Mapping[str, float],
    representation: str = "domain_scores",
    n_profiles: int | None = None,
    indeterminate_mass: float = 0.0,
    model_version: str = "0.1.0",
    extra_warnings: list[str] | None = None,
) -> PhenotypeProfile:
    """Assemble the participant's soft profile, reserving indeterminate mass.

    ``indeterminate_mass`` is supplied by the abstention layer: a fully abstaining
    participant gets 1.0 and therefore no named dominant profile at all.
    """
    warnings = list(extra_warnings or [])
    probabilities = dict(membership)
    if indeterminate_mass > 0:
        probabilities = add_indeterminate_mass(probabilities, indeterminate_mass)

    dominant, dominant_probability = dominant_with_indeterminate(probabilities)

    for score in domain_scores.values():
        warnings.extend(score.warnings)

    return PhenotypeProfile(
        patient_id=patient_id,
        domain_scores=dict(domain_scores),
        phenotype_probabilities=probabilities,
        dominant_profile=dominant,
        dominant_probability=float(dominant_probability),
        representation=representation,
        n_profiles=n_profiles if n_profiles is not None else len(membership),
        model_version=model_version,
        warnings=warnings,
    )
