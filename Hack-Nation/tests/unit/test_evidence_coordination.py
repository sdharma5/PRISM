"""Invariants of the evidence coordination and PMOS interpretation layer.

These tests guard the claims the layer is allowed to make. Most of them assert
that something is *refused*, because the failure mode here is not a crash -- it
is a plausible-looking number that overstates what the system knows.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference import EvidenceCoordinator, coordinate_only
from inference.disagreement import classify_agreement, explain_disagreement
from inference.domain_mapper import map_token_to_domains
from inference.report_schema import DomainEvidence, PatientEvidenceReport
from models.adapters.pmos.evidence_adapter import PmosEvidenceAdapter
from models.adapters.pmos.profile_output import PMOSProfileOutput
from schemas.modality_token import ModalityToken


def static_token(**features: float) -> ModalityToken:
    base = {
        "reproductive_score": 1.5,
        "androgenic_score": 0.8,
        "metabolic_score": 0.3,
        "ovarian_score": 1.0,
        "cycle_length": 48.0,
        "hirsutism": True,
        "anti_mullerian_hormone": 7.0,
    }
    base.update(features)
    return ModalityToken(
        patient_id="P1",
        modality="static_clinical",
        structured_features=base,
        quality_score=0.9,
        confidence_score=0.85,
    )


def ultrasound_token(count: float = 24.0) -> ModalityToken:
    return ModalityToken(
        patient_id="P1",
        modality="ovarian_ultrasound",
        structured_features={"follicle_number_per_ovary": count, "ovary_volume_ml": 11.0},
        quality_score=0.8,
        confidence_score=0.75,
    )


def temporal_token(irregularity: float = 0.2) -> ModalityToken:
    return ModalityToken(
        patient_id="P1",
        modality="longitudinal_hormonal_state",
        structured_features={"cycle_irregularity": irregularity, "cycle_phase_entropy": 0.6},
        quality_score=0.8,
        confidence_score=0.7,
    )


class FakeHead:
    def predict_proba_from_features(self, values: dict) -> float:  # noqa: ARG002
        return 0.78


# -- coordination ----------------------------------------------------------


def test_tokens_from_different_patients_are_refused() -> None:
    """Coordinating two people would produce one report describing both."""
    other = ultrasound_token()
    other.patient_id = "P2"
    with pytest.raises(ValueError, match="different patients"):
        coordinate_only([static_token(), other])


def test_calibrated_mode_is_unavailable() -> None:
    """No matched validation data exists to fit calibrated weights."""
    with pytest.raises(ValueError, match="matched"):
        EvidenceCoordinator().combine([static_token()], mode="calibrated")


def test_missing_modalities_are_reported_not_imputed() -> None:
    evidence = coordinate_only([static_token()])
    assert evidence.available_modalities == ["static_clinical"]
    assert "ovarian_ultrasound" in evidence.missing_modalities
    assert evidence.coverage == pytest.approx(1 / 3)


def test_domain_with_no_contributor_abstains() -> None:
    """A domain no available modality can speak to yields no number."""
    evidence = coordinate_only([ultrasound_token()])
    metabolic = evidence.domain_evidence["metabolic"]
    assert metabolic.level == "insufficient_evidence"
    assert metabolic.score is None


def test_zero_weight_modality_never_contributes() -> None:
    """Ultrasound has weight 0.0 for metabolic and must be excluded entirely."""
    evidence = coordinate_only([static_token(), ultrasound_token()])
    assert "ovarian_ultrasound" not in evidence.domain_evidence["metabolic"].supporting_modalities


def test_separate_mode_computes_no_combined_score() -> None:
    evidence = EvidenceCoordinator().combine(
        [static_token(), ultrasound_token()], mode="separate"
    )
    morphology = evidence.domain_evidence["ovarian_morphology"]
    assert morphology.level == "not_combined"
    assert morphology.score is None
    # Per-modality readings survive so each encoder is still visible.
    assert len(morphology.modality_scores) == 2


# -- disagreement ----------------------------------------------------------


def test_conflicting_sources_are_flagged_not_averaged_silently() -> None:
    """A wide spread must surface a note, not vanish into the mean."""
    evidence = coordinate_only([static_token(reproductive_score=2.5), temporal_token(0.05)])
    reproductive = evidence.domain_evidence["reproductive"]
    assert reproductive.agreement == "conflicting"
    assert reproductive.notes, "a conflicting domain must carry an explanation"
    assert "too short" in reproductive.notes[0]


def test_agreement_classification_bands() -> None:
    assert classify_agreement({}) == "none"
    assert classify_agreement({"a": 0.5}) == "single_source"
    assert classify_agreement({"a": 0.50, "b": 0.55}) == "strong"
    assert classify_agreement({"a": 0.50, "b": 0.70}) == "moderate"
    assert classify_agreement({"a": 0.10, "b": 0.90}) == "conflicting"


def test_agreeing_sources_produce_no_disagreement_note() -> None:
    assert explain_disagreement("reproductive", {"a": 0.50, "b": 0.55}) is None


# -- domain mapping --------------------------------------------------------


def test_unknown_modality_raises_rather_than_contributing_nothing() -> None:
    """A speech token reaching the coordinator is a wiring error, not a no-op."""
    speech = ModalityToken(
        patient_id="P1", modality="speech_symptoms", quality_score=0.5, confidence_score=0.5
    )
    with pytest.raises(ValueError, match="ingestion|static clinical encoder"):
        map_token_to_domains(speech)


def test_pcom_threshold_anchors_the_ramp() -> None:
    """Exactly at the guideline threshold the evidence is equivocal."""
    scores, _ = map_token_to_domains(
        ModalityToken(
            patient_id="P1",
            modality="ovarian_ultrasound",
            structured_features={"follicle_number_per_ovary": 20.0},
            quality_score=1.0,
            confidence_score=1.0,
        )
    )
    assert scores["ovarian_morphology"] == pytest.approx(0.5, abs=1e-6)


def test_per_section_count_is_not_treated_as_per_ovary() -> None:
    """A single-frame count must never be compared to the PCOM threshold."""
    scores, _ = map_token_to_domains(
        ModalityToken(
            patient_id="P1",
            modality="ovarian_ultrasound",
            structured_features={"follicle_number_per_section": 25.0},
            quality_score=1.0,
            confidence_score=1.0,
        )
    )
    assert "ovarian_morphology" not in scores


# -- adapter ---------------------------------------------------------------


def test_ultrasound_alone_cannot_produce_a_pmos_probability() -> None:
    profile = PmosEvidenceAdapter().predict(coordinate_only([ultrasound_token()]))
    assert profile.abstain is True
    assert profile.pmos_evidence_probability is None


def test_full_evidence_with_learned_head_produces_a_probability() -> None:
    profile = PmosEvidenceAdapter(static_model=FakeHead()).predict(
        coordinate_only([static_token(), ultrasound_token()])
    )
    assert profile.abstain is False
    assert profile.pmos_evidence_probability == pytest.approx(0.78)
    assert "static_clinical.pmos_head" in profile.learned_components_used
    assert profile.rule_based_components_used


def test_guideline_met_axis_reports_high() -> None:
    profile = PmosEvidenceAdapter(static_model=FakeHead()).predict(
        coordinate_only([static_token(), ultrasound_token(count=24.0)])
    )
    morphology = profile.diagnostic_feature_evidence["polycystic_ovarian_morphology"]
    assert morphology.axis_status == "met"
    assert morphology.level == "high"


def test_conflicting_variables_are_recorded_not_reconciled() -> None:
    """Two modalities reporting one variable differently must surface a conflict."""
    clashing = ultrasound_token()
    clashing.structured_features["anti_mullerian_hormone"] = 2.0
    profile = PmosEvidenceAdapter(static_model=FakeHead()).predict(
        coordinate_only([static_token(), clashing])
    )
    assert any("anti_mullerian_hormone" in c or "reported" in c for c in profile.conflicts)


# -- schema guards ---------------------------------------------------------


def test_abstention_forbids_reporting_the_probability() -> None:
    with pytest.raises(ValidationError):
        PMOSProfileOutput(
            patient_id="P1",
            abstain=True,
            pmos_evidence_probability=0.7,
            rule_based_components_used=["x"],
        )


def test_probability_requires_the_learned_head() -> None:
    """No rule-based component may issue a whole-patient PMOS probability."""
    with pytest.raises(ValidationError, match="learned static"):
        PMOSProfileOutput(
            patient_id="P1",
            pmos_evidence_probability=0.7,
            rule_based_components_used=["pmos_adapter.guideline_axis_thresholds"],
        )


def test_profile_must_declare_its_method() -> None:
    with pytest.raises(ValidationError, match="declare"):
        PMOSProfileOutput(patient_id="P1")


def test_joint_model_claim_is_unreachable() -> None:
    """Nothing in this repository may claim a jointly trained multimodal model."""
    with pytest.raises(ValidationError, match="ADR-002|not reachable"):
        PatientEvidenceReport(patient_id="P1", joint_model_used=True)


def test_insufficient_evidence_forbids_a_score() -> None:
    with pytest.raises(ValidationError):
        DomainEvidence(domain="metabolic", level="insufficient_evidence", score=0.5)
