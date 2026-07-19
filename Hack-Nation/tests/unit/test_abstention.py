"""Every indeterminate rule must fire when it should — and only when it should.

Each rule gets a positive test (the triggering condition alone produces that
reason) and the shared ``clean_evidence`` fixture asserts the negative case: a
participant who passes every rule must not be abstained on. A silently
never-firing rule and a rule that fires on everything are equally useless.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.phenotype.indeterminate import (
    add_indeterminate_mass,
    dominant_with_indeterminate,
    is_indeterminate,
)
from models.stability.abstention import (
    REASON_CODES,
    AbstentionEvidence,
    AbstentionThresholds,
    evaluate_abstention,
    scaled_distance_percentile,
)
from schemas.phenotype import INDETERMINATE
from tests.fixtures.synthetic_clusters import DEFAULT_FEATURES, make_far_outlier


@pytest.fixture
def clean_evidence() -> AbstentionEvidence:
    """A participant who passes all six rules comfortably."""
    return AbstentionEvidence(
        patient_id="p1",
        dominant_profile="profile_0",
        probabilities={"profile_0": 0.85, "profile_1": 0.10, "profile_2": 0.05},
        alternative_assignments=["profile_0"] * 5,
        bootstrap_agreement=0.95,
        cluster_jaccard=0.90,
        flipping_features=[],
        n_features_tested=8,
        fragility_by_feature={"bmi": 0.02, "shbg": 0.05},
        distance_percentile=0.40,
        n_defining_features_observed=8,
        n_defining_features_total=8,
        subtype_flip_rate=0.02,
    )


def _reasons(report) -> str:
    return " | ".join(report.abstain_reasons)


# ------------------------------------------------------- the negative control


def test_no_abstention_when_every_rule_is_satisfied(clean_evidence):
    report = evaluate_abstention(clean_evidence)
    assert report.abstain is False
    assert report.abstain_reasons == []
    assert report.dominant_profile == "profile_0"
    assert report.dominant_probability == pytest.approx(0.85)
    assert report.stability_score > 0.9


# ------------------------------------------------------------ rule by rule


def test_rule1_low_confidence_fires(clean_evidence):
    evidence = clean_evidence
    evidence.probabilities = {"profile_0": 0.40, "profile_1": 0.35, "profile_2": 0.25}
    report = evaluate_abstention(evidence)
    assert report.abstain
    assert "low_confidence" in _reasons(report)


def test_rule1_does_not_fire_just_above_threshold(clean_evidence):
    clean_evidence.probabilities = {"profile_0": 0.51, "profile_1": 0.49}
    report = evaluate_abstention(clean_evidence)
    assert "low_confidence" not in _reasons(report)


def test_rule2_model_disagreement_fires(clean_evidence):
    clean_evidence.alternative_assignments = [
        "profile_1",
        "profile_2",
        "profile_1",
        "profile_0",
        "profile_2",
    ]
    report = evaluate_abstention(clean_evidence)
    assert report.abstain
    assert "model_disagreement" in _reasons(report)


def test_rule2_does_not_fire_when_models_mostly_agree(clean_evidence):
    clean_evidence.alternative_assignments = ["profile_0", "profile_0", "profile_0", "profile_1"]
    assert "model_disagreement" not in _reasons(evaluate_abstention(clean_evidence))


def test_rule3a_unstable_bootstrap_assignment_fires(clean_evidence):
    clean_evidence.bootstrap_agreement = 0.35
    report = evaluate_abstention(clean_evidence)
    assert report.abstain
    assert "unstable_bootstrap_assignment" in _reasons(report)


def test_rule3b_unstable_cluster_jaccard_fires(clean_evidence):
    clean_evidence.cluster_jaccard = 0.41
    report = evaluate_abstention(clean_evidence)
    assert report.abstain
    assert "does not reliably reappear" in _reasons(report)


def test_rule4_single_variable_fragility_fires(clean_evidence):
    clean_evidence.flipping_features = ["anti_mullerian_hormone"]
    report = evaluate_abstention(clean_evidence)
    assert report.abstain
    assert "single_variable_fragility" in _reasons(report)
    assert "anti_mullerian_hormone" in _reasons(report)


def test_rule4_can_be_relaxed_to_a_fraction(clean_evidence):
    clean_evidence.flipping_features = ["bmi"]
    thresholds = AbstentionThresholds(
        abstain_on_any_single_feature_flip=False, max_fragile_feature_fraction=0.5
    )
    assert "single_variable_fragility" not in _reasons(
        evaluate_abstention(clean_evidence, thresholds)
    )


def test_rule5_far_from_all_profiles_fires(clean_evidence):
    clean_evidence.distance_percentile = 0.99
    report = evaluate_abstention(clean_evidence)
    assert report.abstain
    assert "far_from_all_profiles" in _reasons(report)


def test_rule6_insufficient_observed_evidence_fires(clean_evidence):
    clean_evidence.n_defining_features_observed = 2
    clean_evidence.n_defining_features_total = 8
    report = evaluate_abstention(clean_evidence)
    assert report.abstain
    assert "insufficient_observed_evidence" in _reasons(report)


def test_rule6_does_not_fire_at_adequate_coverage(clean_evidence):
    clean_evidence.n_defining_features_observed = 5
    clean_evidence.n_defining_features_total = 8
    assert "insufficient_observed_evidence" not in _reasons(evaluate_abstention(clean_evidence))


def test_every_documented_reason_code_is_reachable(clean_evidence):
    """No rule may be dead code."""
    triggers = {
        "low_confidence": {"probabilities": {"profile_0": 0.3, "profile_1": 0.7}},
        "model_disagreement": {"alternative_assignments": ["profile_1"] * 5},
        "unstable_bootstrap_assignment": {"bootstrap_agreement": 0.1},
        "single_variable_fragility": {"flipping_features": ["bmi"]},
        "far_from_all_profiles": {"distance_percentile": 0.99},
        "insufficient_observed_evidence": {"n_defining_features_observed": 1},
    }
    for code, mutation in triggers.items():
        evidence = AbstentionEvidence(**{**clean_evidence.__dict__, **mutation})
        assert code in _reasons(evaluate_abstention(evidence)), code
    assert set(triggers) == set(REASON_CODES)


# -------------------------------------------------------- unchecked rules warn


def test_missing_evidence_produces_warnings_not_silent_passes():
    evidence = AbstentionEvidence(
        patient_id="p2",
        dominant_profile="profile_0",
        probabilities={"profile_0": 0.9, "profile_1": 0.1},
    )
    report = evaluate_abstention(evidence)
    assert report.abstain is False
    joined = " ".join(report.warnings)
    for fragment in (
        "model_disagreement check not performed",
        "bootstrap stability check not performed",
        "single_variable_fragility check not performed",
        "distance_to_profiles check not performed",
        "evidence_coverage check not performed",
    ):
        assert fragment in joined


def test_multiple_rules_can_fire_at_once(clean_evidence):
    clean_evidence.probabilities = {"profile_0": 0.3, "profile_1": 0.4, "profile_2": 0.3}
    clean_evidence.bootstrap_agreement = 0.2
    clean_evidence.distance_percentile = 0.99
    report = evaluate_abstention(clean_evidence)
    assert len(report.abstain_reasons) >= 3


def test_high_entropy_warns_without_being_its_own_abstention(clean_evidence):
    clean_evidence.probabilities = {"profile_0": 0.52, "profile_1": 0.48}
    report = evaluate_abstention(clean_evidence)
    assert "entropy" in " ".join(report.warnings)


# ------------------------------------------------------------ distance helper


def test_scaled_distance_percentile_ranks_an_outlier_at_the_top():
    rng = np.random.default_rng(0)
    cohort = np.vstack(
        [
            rng.normal(-2, 0.4, size=(30, len(DEFAULT_FEATURES))),
            rng.normal(2, 0.4, size=(30, len(DEFAULT_FEATURES))),
        ]
    )
    centers = np.vstack([cohort[:30].mean(axis=0), cohort[30:].mean(axis=0)])
    outlier = np.array(list(make_far_outlier().values()))
    assert scaled_distance_percentile(outlier, centers, cohort) >= 0.99
    assert scaled_distance_percentile(cohort[0], centers, cohort) < 0.95


# ----------------------------------------------------------- indeterminate mass


def test_add_indeterminate_mass_renormalizes_and_preserves_ratios():
    out = add_indeterminate_mass({"a": 0.6, "b": 0.3, "c": 0.1}, 0.5)
    assert sum(out.values()) == pytest.approx(1.0)
    assert out[INDETERMINATE] == pytest.approx(0.5)
    assert out["a"] / out["b"] == pytest.approx(2.0)


def test_full_indeterminate_mass_zeroes_the_profiles():
    out = add_indeterminate_mass({"a": 0.9, "b": 0.1}, 1.0)
    assert out[INDETERMINATE] == 1.0
    assert out["a"] == 0.0
    assert is_indeterminate(out)
    assert dominant_with_indeterminate(out) == (INDETERMINATE, 1.0)


def test_empty_or_zero_input_collapses_to_indeterminate():
    assert add_indeterminate_mass({}, 0.0)[INDETERMINATE] == 1.0
    assert add_indeterminate_mass({"a": 0.0}, 0.0)[INDETERMINATE] == 1.0


def test_exact_tie_resolves_to_indeterminate():
    assert dominant_with_indeterminate({"a": 0.5, INDETERMINATE: 0.5})[0] == INDETERMINATE
