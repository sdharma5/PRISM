"""Stability, sensitivity, and the affinity-vs-probability distinction."""

from __future__ import annotations

import pytest

from models.adapters.pcos.prototype_similarity import PrototypeSimilarityModel
from models.adapters.pcos.stability import PhenotypeStabilityEngine

CLEAR_METABOLIC = {
    "metabolic": 2.0,
    "symptom_burden": 0.8,
    "androgenic": 0.4,
    "lh_amh_pattern": -1.2,
    "ovarian": -0.3,
    "reproductive": 0.1,
}

BORDERLINE = {
    "metabolic": 0.30,
    "symptom_burden": 0.28,
    "androgenic": 0.31,
    "lh_amh_pattern": 0.29,
    "ovarian": 0.30,
    "reproductive": 0.30,
}

THIN = {"metabolic": 1.0, "reproductive": None, "androgenic": None, "ovarian": None}


def model() -> PrototypeSimilarityModel:
    return PrototypeSimilarityModel()


def engine(**kwargs) -> PhenotypeStabilityEngine:
    return PhenotypeStabilityEngine(n_bootstrap=60, seed=0, **kwargs)


# -- affinities are not probabilities --------------------------------------


def test_affinities_are_named_affinities_not_probabilities() -> None:
    """The field name carries the semantics; renaming it back would mislead."""
    result = model().predict(CLEAR_METABOLIC)
    assert hasattr(result, "affinities")
    assert not hasattr(result, "probabilities")


def test_affinity_magnitude_depends_on_an_arbitrary_temperature() -> None:
    """Demonstrates why affinities must not be read as calibrated probabilities."""
    sharp = PrototypeSimilarityModel(temperature=0.05).predict(CLEAR_METABOLIC)
    soft = PrototypeSimilarityModel(temperature=1.0).predict(CLEAR_METABOLIC)
    assert max(sharp.affinities.values()) > max(soft.affinities.values()) + 0.1


# -- minimum observed domains ----------------------------------------------


def test_thin_evidence_abstains_from_any_profile() -> None:
    report = engine(min_observed_domains=3).evaluate(THIN, model())
    assert report.meets_minimum_domains is False
    assert report.abstain_from_profile is True
    assert "at least 3" in (report.abstention_reason or "")


def test_thin_evidence_skips_bootstrap_entirely() -> None:
    """A bootstrap over 1 domain would report confident agreement about nothing."""
    report = engine(min_observed_domains=3).evaluate(THIN, model())
    assert report.n_bootstrap == 0
    assert report.bootstrap_agreement == 0.0


# -- perturbations ---------------------------------------------------------


def test_clear_pattern_is_more_stable_than_borderline_one() -> None:
    clear = engine().evaluate(CLEAR_METABOLIC, model())
    borderline = engine().evaluate(BORDERLINE, model())
    assert clear.bootstrap_agreement > borderline.bootstrap_agreement


def test_domain_ablation_is_reported_per_domain() -> None:
    report = engine().evaluate(CLEAR_METABOLIC, model())
    assert set(report.domain_ablation) == {k for k, v in CLEAR_METABOLIC.items() if v is not None}


def test_modality_removal_is_evaluated() -> None:
    report = engine().evaluate(
        CLEAR_METABOLIC, model(), available_modalities=["static_clinical", "ovarian_ultrasound"]
    )
    assert set(report.modality_removal) == {"static_clinical", "ovarian_ultrasound"}


def test_temperature_and_threshold_sensitivity_are_populated() -> None:
    report = engine().evaluate(CLEAR_METABOLIC, model())
    assert "dominant_by_temperature" in report.temperature_sensitivity
    assert "dominant_by_threshold" in report.threshold_sensitivity
    assert isinstance(report.temperature_sensitivity["dominant_is_stable"], bool)


def test_stability_score_collapses_when_any_factor_collapses() -> None:
    """Product, not mean: one fragile axis should not be averaged away."""
    borderline = engine().evaluate(BORDERLINE, model())
    assert borderline.stability_score <= borderline.bootstrap_agreement


def test_unstable_assignment_produces_a_warning() -> None:
    report = engine(min_bootstrap_agreement=0.99).evaluate(BORDERLINE, model())
    assert report.is_stable is False
    assert any("unstable" in w.lower() for w in report.warnings)


# -- adapter integration ---------------------------------------------------


def test_adapter_withholds_dominant_profile_when_unstable() -> None:
    """An unstable label is the part a reader would quote, so it is withdrawn."""
    from inference import coordinate_only
    from models.adapters.pcos.evidence_adapter import PcosEvidenceAdapter
    from schemas.modality_token import ModalityToken

    token = ModalityToken(
        patient_id="P1",
        modality="static_clinical",
        structured_features={f"{k}_score": v for k, v in BORDERLINE.items()},
        quality_score=0.9,
        confidence_score=0.8,
    )
    out = PcosEvidenceAdapter(
        prototype_model=model(),
        stability_engine=engine(min_bootstrap_agreement=0.99),
    ).predict(coordinate_only([token]))

    assert out.dominant_profile is None
    # The affinities stay visible: hiding them would conceal that the patient
    # sits between profiles.
    assert out.phenotype_affinities
    assert out.profile_stability["available"] is True
    assert out.profile_stability["is_stable"] is False


def test_stability_payload_records_absence_when_no_engine() -> None:
    from inference import coordinate_only
    from models.adapters.pcos.evidence_adapter import PcosEvidenceAdapter
    from schemas.modality_token import ModalityToken

    token = ModalityToken(
        patient_id="P1",
        modality="static_clinical",
        structured_features={f"{k}_score": v for k, v in CLEAR_METABOLIC.items()},
        quality_score=0.9,
        confidence_score=0.8,
    )
    out = PcosEvidenceAdapter(prototype_model=model()).predict(coordinate_only([token]))
    assert out.profile_stability["available"] is False
    assert "provisional" in out.profile_stability["reason"]


def test_stability_is_deterministic_for_a_fixed_seed() -> None:
    first = engine().evaluate(CLEAR_METABOLIC, model())
    second = engine().evaluate(CLEAR_METABOLIC, model())
    assert first.bootstrap_agreement == pytest.approx(second.bootstrap_agreement)
