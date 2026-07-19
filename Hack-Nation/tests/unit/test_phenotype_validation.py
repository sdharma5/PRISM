"""Cohort-level validation sections in scripts/validate_phenotype_profiles.py.

These guard the sensitivity analyses that decide whether a reported phenotype
profile is a finding or an artifact of a cut-point we chose.
"""

from __future__ import annotations

import pytest

from models.adapters.pcos.prototype_similarity import PrototypeSimilarityModel
from scripts.validate_phenotype_profiles import (
    defining_domain_support,
    minimum_observed_domain_rule,
    temperature_sensitivity,
    threshold_sensitivity,
)

DOMAINS = (
    "reproductive",
    "metabolic",
    "clinical_androgenic_evidence",
    "biochemical_androgenic_evidence",
    "ovarian",
    "lh_amh_pattern",
    "symptom_burden",
)


def row(**scores: float | None) -> dict[str, float | None]:
    """A domain-score row, defaulting every unnamed domain to unobserved."""
    return {domain: scores.get(domain) for domain in DOMAINS}


@pytest.fixture
def cohort() -> list[dict[str, float | None]]:
    """A small cohort in which NEITHER androgenic domain is ever observed.

    Both halves absent is the case that must make `androgenic_leaning`
    unreachable; a cohort missing only the assays would still leave the profile
    legitimately available on clinical signs.
    """
    return [
        row(metabolic=1.8, reproductive=0.9, ovarian=0.4, lh_amh_pattern=0.3, symptom_burden=0.5),
        row(metabolic=1.5, reproductive=0.7, ovarian=0.2, lh_amh_pattern=0.4, symptom_burden=0.6),
        row(metabolic=0.2, reproductive=1.1, ovarian=1.6, lh_amh_pattern=1.9, symptom_burden=0.4),
        row(metabolic=0.1, reproductive=1.3, ovarian=1.4, lh_amh_pattern=2.0, symptom_burden=0.3),
        row(metabolic=0.6, reproductive=0.6, ovarian=0.5, lh_amh_pattern=0.6, symptom_burden=1.9),
    ]


# -- temperature -----------------------------------------------------------


def test_temperature_reports_every_swept_value(cohort: list[dict[str, float | None]]) -> None:
    result = temperature_sensitivity(cohort, temperatures=(0.10, 0.25, 1.00))
    assert set(result["by_temperature"]) == {"0.10", "0.25", "1.00"}


def test_baseline_temperature_flips_nothing_against_itself(
    cohort: list[dict[str, float | None]],
) -> None:
    result = temperature_sensitivity(cohort, temperatures=(0.25,), baseline=0.25)
    assert result["by_temperature"]["0.25"]["dominant_flip_rate_vs_baseline"] == 0.0


def test_higher_temperature_softens_the_affinities(
    cohort: list[dict[str, float | None]],
) -> None:
    """Temperature controls sharpness, which is the whole reason it is swept."""
    result = temperature_sensitivity(cohort, temperatures=(0.10, 1.00))
    sharp = result["by_temperature"]["0.10"]
    soft = result["by_temperature"]["1.00"]
    assert soft["mean_entropy"] > sharp["mean_entropy"]
    assert soft["mean_top_affinity"] < sharp["mean_top_affinity"]


# -- thresholds ------------------------------------------------------------


def test_threshold_grid_is_the_full_cross_product(
    cohort: list[dict[str, float | None]],
) -> None:
    result = threshold_sensitivity(cohort, margins=(0.05, 0.10), floors=(0.20, 0.30))
    assert len(result["grid"]) == 4


def test_baseline_threshold_cell_flips_nothing(cohort: list[dict[str, float | None]]) -> None:
    result = threshold_sensitivity(
        cohort, margins=(0.10,), floors=(0.30,), baseline_margin=0.10, baseline_floor=0.30
    )
    cell = result["grid"]["margin=0.10,floor=0.30"]
    assert cell["dominant_flip_rate_vs_baseline"] == 0.0


def test_raising_the_similarity_floor_never_reduces_indeterminacy(
    cohort: list[dict[str, float | None]],
) -> None:
    """A stricter match requirement can only send more patients to indeterminate."""
    result = threshold_sensitivity(cohort, margins=(0.10,), floors=(0.20, 0.40))
    low = result["grid"]["margin=0.10,floor=0.20"]["indeterminate_fraction"]
    high = result["grid"]["margin=0.10,floor=0.40"]["indeterminate_fraction"]
    assert high >= low


# -- minimum observed domains ---------------------------------------------


def test_never_observed_domain_is_reported(cohort: list[dict[str, float | None]]) -> None:
    result = minimum_observed_domain_rule(cohort)
    assert result["never_observed_domains"] == [
        "biochemical_androgenic_evidence",
        "clinical_androgenic_evidence",
    ]
    assert result["per_domain_observation_rate"]["clinical_androgenic_evidence"] == 0.0
    assert result["per_domain_observation_rate"]["biochemical_androgenic_evidence"] == 0.0


def test_floor_counts_patients_it_would_exclude() -> None:
    thin = [row(metabolic=1.0, reproductive=0.5), row(metabolic=1.0)]
    result = minimum_observed_domain_rule(thin, floors=(2, 3))
    assert result["by_floor"]["2"]["n_below_floor"] == 1
    assert result["by_floor"]["3"]["n_below_floor"] == 2


def test_patients_below_the_floor_are_all_indeterminate() -> None:
    """The rule refuses to place a patient rather than guessing from one axis."""
    thin = [row(metabolic=1.0)]
    result = minimum_observed_domain_rule(thin, floors=(3,))
    assert result["by_floor"]["3"]["indeterminate_fraction"] == 1.0


# -- defining-domain support ----------------------------------------------


def test_profile_matched_without_its_defining_domain_is_flagged(
    cohort: list[dict[str, float | None]],
) -> None:
    """The androgenic prototype must not claim support it never measured.

    Previously this profile was assigned on secondary weights alone and the
    resulting 0.0 support fraction was reported as a caveat. The eligibility
    gate now makes the assignment impossible, so the assertion is that nobody
    receives the label at all.
    """
    model = PrototypeSimilarityModel()
    result = defining_domain_support(cohort, model)
    androgenic = result["by_profile"]["androgenic_leaning"]

    assert androgenic["defining_domains"] == [
        "clinical_androgenic_evidence",
        "biochemical_androgenic_evidence",
    ]
    assert androgenic["n_assigned"] == 0
    assert result["violations"] == []


def test_observed_defining_domain_reports_full_support(
    cohort: list[dict[str, float | None]],
) -> None:
    model = PrototypeSimilarityModel()
    result = defining_domain_support(cohort, model)
    for entry in result["by_profile"].values():
        if not entry["n_assigned"]:
            continue
        assert entry["fraction_with_defining_domain_observed"] == 1.0
