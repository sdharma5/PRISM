"""End-to-end PCOS profile behaviour across the modality combinations a real patient may present with.

Each scenario asserts a *claim boundary* rather than a specific number. The
numbers depend on models that will be retrained; the boundaries -- what the
system is allowed to assert given what it was given -- must not move.
"""

from __future__ import annotations

import pytest

from inference import coordinate_only
from models.adapters.pcos.evidence_adapter import PcosEvidenceAdapter
from models.adapters.pcos.prototype_similarity import PrototypeSimilarityModel
from schemas.modality_token import ModalityToken


class FakeStaticHead:
    """Stands in for the trained logistic head so tests need no checkpoint."""

    def predict_proba_from_features(self, values: dict) -> float:  # noqa: ARG002
        return 0.78


def static_token(**overrides) -> ModalityToken:
    features = {
        "reproductive_score": 1.4,
        "metabolic_score": 0.6,
        "clinical_androgenic_evidence_score": 1.1,
        "biochemical_androgenic_evidence_score": 0.9,
        "ovarian_score": 0.9,
        "lh_amh_pattern_score": 0.8,
        "symptom_burden_score": 1.2,
        "cycle_length": 48.0,
        "hirsutism": True,
        "anti_mullerian_hormone": 7.4,
        "bmi": 29.0,
    }
    features.update(overrides)
    return ModalityToken(
        patient_id="P1",
        modality="static_clinical",
        structured_features=features,
        quality_score=0.9,
        confidence_score=0.85,
    )


def ultrasound_token(count: float = 23.0) -> ModalityToken:
    return ModalityToken(
        patient_id="P1",
        modality="ovarian_ultrasound",
        structured_features={"follicle_number_per_ovary": count, "ovary_volume_ml": 11.2},
        quality_score=0.85,
        confidence_score=0.8,
    )


def temporal_token(irregularity: float = 0.7) -> ModalityToken:
    return ModalityToken(
        patient_id="P1",
        modality="longitudinal_hormonal_state",
        structured_features={
            "cycle_irregularity": irregularity,
            "cycle_phase_entropy": 0.5,
            "predicted_cycle_phase": "luteal",
        },
        quality_score=0.8,
        confidence_score=0.75,
    )


def adapter(*, with_head: bool = True) -> PcosEvidenceAdapter:
    return PcosEvidenceAdapter(
        static_model=FakeStaticHead() if with_head else None,
        prototype_model=PrototypeSimilarityModel(),
    )


def profile_for(tokens, *, with_head: bool = True):
    return adapter(with_head=with_head).predict(coordinate_only(tokens))


# -- modality combinations -------------------------------------------------


def test_static_only_produces_a_probability_and_a_profile() -> None:
    out = profile_for([static_token()])
    assert out.abstain is False
    assert out.pcos_evidence_probability == pytest.approx(0.78)
    assert "ovarian_ultrasound" in out.missing_modalities
    assert out.phenotype_affinities


def test_static_plus_ultrasound_meets_morphology_axis() -> None:
    out = profile_for([static_token(), ultrasound_token(count=23)])
    assert out.diagnostic_feature_evidence["polycystic_ovarian_morphology"].axis_status == "met"
    assert out.pcos_evidence_probability is not None


def test_static_plus_temporal_reports_current_state() -> None:
    out = profile_for([static_token(), temporal_token()])
    assert "longitudinal_hormonal_state" in out.available_modalities
    assert out.pcos_evidence_probability is not None


def test_all_three_branches_available() -> None:
    out = profile_for([static_token(), ultrasound_token(), temporal_token()])
    assert set(out.available_modalities) == {
        "static_clinical",
        "ovarian_ultrasound",
        "longitudinal_hormonal_state",
    }
    assert out.missing_modalities == []


# -- the abstention boundary (spec item 13) --------------------------------


def test_ultrasound_only_abstains_from_pcos_probability() -> None:
    out = profile_for([ultrasound_token()], with_head=False)
    assert out.abstain is True
    assert out.pcos_evidence_probability is None


def test_temporal_only_abstains_from_pcos_probability() -> None:
    out = profile_for([temporal_token()], with_head=False)
    assert out.abstain is True
    assert out.pcos_evidence_probability is None


def test_ultrasound_only_still_reports_morphology_evidence() -> None:
    """Abstaining from the whole-patient claim must not discard what the scan showed."""
    evidence = coordinate_only([ultrasound_token(count=24)])
    morphology = evidence.domain_evidence["ovarian_morphology"]
    assert morphology.score is not None
    assert "ovarian_ultrasound" in morphology.supporting_modalities


# -- evidence quality ------------------------------------------------------


def test_incomplete_androgen_data_yields_no_biochemical_axis() -> None:
    """A missing assay must leave the axis not_assessable, never 'not met'."""
    out = profile_for([static_token(biochemical_androgenic_evidence_score=None)])
    axis = out.diagnostic_feature_evidence["hyperandrogenism_biochemical"]
    assert axis.axis_status == "not_assessable"
    assert axis.missing_evidence


def test_absent_domain_score_is_none_not_zero() -> None:
    """A z-score of 0.0 means 'average'; None means 'not measured'."""
    out = profile_for([static_token(biochemical_androgenic_evidence_score=None)])
    assert out.phenotype_domain_scores.get("biochemical_androgenic_evidence") is None
    assert out.domain_assessability["biochemical_androgenic_evidence"] is False
    # Absent assays must not drag the observed cutaneous signs down with them.
    assert out.phenotype_domain_scores.get("clinical_androgenic_evidence") is not None
    assert out.androgenic_evidence_source == "symptoms_only"


def test_thin_domain_evidence_gives_indeterminate_profile() -> None:
    out = profile_for(
        [
            static_token(
                metabolic_score=None,
                clinical_androgenic_evidence_score=None,
                biochemical_androgenic_evidence_score=None,
                ovarian_score=None,
                lh_amh_pattern_score=None,
                symptom_burden_score=None,
            )
        ]
    )
    assert out.dominant_profile is None
    assert out.indeterminate is True
    assert out.indeterminate_reasons


def test_conflicting_static_and_temporal_reproductive_evidence_is_preserved() -> None:
    """A long history and a short window that disagree must not be averaged silently."""
    evidence = coordinate_only([static_token(reproductive_score=2.6), temporal_token(0.03)])
    reproductive = evidence.domain_evidence["reproductive"]
    assert reproductive.agreement == "conflicting"
    assert reproductive.notes

    out = adapter().predict(evidence)
    assert out.conflicts


# -- claim discipline ------------------------------------------------------


def test_profile_always_declares_learned_versus_rule_based() -> None:
    out = profile_for([static_token(), ultrasound_token()])
    assert "static_clinical.pcos_head" in out.learned_components_used
    assert any("guideline" in c for c in out.rule_based_components_used)


def test_phenotype_output_is_never_described_as_a_subtype() -> None:
    out = profile_for([static_token()])
    note = out.explanation["phenotype_profile_similarity"]["note"].lower()
    assert "not validated clinical subtypes" in note


def test_explanation_has_every_required_section() -> None:
    out = profile_for([static_token(), ultrasound_token(), temporal_token()])
    for section in (
        "observed_evidence",
        "learned_static_prediction",
        "rotterdam_axis_assessment",
        "phenotype_profile_similarity",
        "ultrasound_morphology_evidence",
        "current_temporal_state_evidence",
        "conflicting_evidence",
        "missing_evidence",
        "uncertainty",
        "abstention",
    ):
        assert section in out.explanation


def test_explanation_attributes_the_probability_to_the_static_head() -> None:
    out = profile_for([static_token(), ultrasound_token()])
    learned = out.explanation["learned_static_prediction"]
    assert learned["available"] is True
    assert "static_clinical.pcos_head" in learned["provenance"]
