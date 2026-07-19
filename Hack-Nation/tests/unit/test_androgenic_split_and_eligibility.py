"""Invariants for the split androgenic axis, profile eligibility, and calibration.

Each test here corresponds to a specific way the previous implementation was
wrong, and is written to fail loudly if that behaviour returns. They are
invariants rather than regression snapshots: none of them asserts a tuned number,
so none can be satisfied by moving a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from evaluation.calibration import (
    ALLOWED_CALIBRATION_FIT_SOURCE,
    DEFAULT_N_BINS,
    PlattCalibrator,
    equal_frequency_bins,
)
from features.phenotype_domains import PhenotypeDomainScorer
from models.adapters.pcos.profile_output import PCOSProfileOutput
from models.adapters.pcos.prototype_similarity import (
    MIXED_MIN_ASSESSABLE_DOMAINS,
    PrototypeSimilarityModel,
    androgenic_evidence_source,
    summarize,
)
from models.adapters.pcos.stability import PhenotypeStabilityEngine
from tests.fixtures.synthetic_tabular import make_synthetic_cohort

ALL_DOMAINS = (
    "reproductive",
    "metabolic",
    "clinical_androgenic_evidence",
    "biochemical_androgenic_evidence",
    "ovarian",
    "lh_amh_pattern",
    "symptom_burden",
)


def row(**scores: float) -> dict[str, float | None]:
    """A domain-score row; every unnamed domain is unobserved, not zero."""
    return {domain: scores.get(domain) for domain in ALL_DOMAINS}


@pytest.fixture
def scorer() -> PhenotypeDomainScorer:
    return PhenotypeDomainScorer().fit(make_synthetic_cohort(n=200, seed=3, missing_rate=0.15))


# -- 1. absent assays must not make symptom evidence unassessable -----------


def test_absent_androgen_assays_leave_clinical_symptom_evidence_assessable(
    scorer: PhenotypeDomainScorer,
) -> None:
    """The original defect, stated directly.

    A patient with three recorded cutaneous signs and no androgen panel must
    have assessable clinical androgenic evidence. Under the merged domain the
    4.5 units of absent assay weight sat in the denominator and pushed the
    observed 1.5 units to 0.19 coverage, below the 0.25 floor -- so the score
    was withheld and the patient was reported as having no androgenic evidence
    while three androgenic signs sat in the record.
    """
    patient = pd.DataFrame([{"acne": 1.0, "androgenic_alopecia": 1.0, "hair_growth_face": 1.0}])
    scored = scorer.score_frame(patient)

    clinical = scored["clinical_androgenic_evidence"][0]
    biochemical = scored["biochemical_androgenic_evidence"][0]

    assert clinical.is_assessable, "observed symptoms must be assessable with no assay drawn"
    assert clinical.score is not None
    assert biochemical.is_assessable is False
    # The clinical denominator must contain no assay weight whatsoever.
    assert {f.code for f in scorer.specs["clinical_androgenic_evidence"].features}.isdisjoint(
        {"total_testosterone", "free_testosterone", "dheas", "shbg"}
    )


def test_androgenic_evidence_source_names_all_four_states() -> None:
    """The combined output must always say which evidence it rests on."""
    assert androgenic_evidence_source(row(clinical_androgenic_evidence=1.0)) == "symptoms_only"
    assert (
        androgenic_evidence_source(row(biochemical_androgenic_evidence=1.0)) == "biochemical_only"
    )
    assert (
        androgenic_evidence_source(
            row(clinical_androgenic_evidence=1.0, biochemical_androgenic_evidence=1.0)
        )
        == "both"
    )
    assert androgenic_evidence_source(row(metabolic=1.0)) == "unavailable"


# -- 2. androgenic_leaning is impossible without androgenic evidence --------


def test_androgenic_leaning_is_unreachable_without_androgenic_evidence() -> None:
    """No arrangement of the other domains may produce the androgenic label.

    The extreme case is the one that matters: a patient whose every non-androgenic
    domain is pushed hard toward the androgenic centroid's secondary weights. If
    the profile is reachable at all, it is reachable here.
    """
    model = PrototypeSimilarityModel()
    adversarial = row(symptom_burden=3.0, reproductive=3.0, metabolic=3.0, ovarian=3.0)

    result = model.predict(adversarial)

    assert "androgenic_leaning" not in result.eligible_profiles
    assert "androgenic_leaning" not in result.similarities
    assert "androgenic_leaning" not in result.affinities
    assert "androgenic_leaning" in result.ineligible_profiles
    assert result.dominant != "androgenic_leaning"


def test_androgenic_leaning_is_eligible_on_clinical_evidence_alone() -> None:
    """Symptoms-only is genuine androgenic evidence, and must not be excluded.

    The eligibility rule exists to stop unsupported labels, not to make the
    profile unreachable in every cohort lacking an assay.
    """
    model = PrototypeSimilarityModel()
    result = model.predict(row(clinical_androgenic_evidence=2.0, symptom_burden=1.5, metabolic=0.2))

    assert "androgenic_leaning" in result.eligible_profiles
    assert result.androgenic_evidence_source == "symptoms_only"


def test_the_output_schema_refuses_the_unsupported_label() -> None:
    """Belt and braces: even a hand-built output cannot carry the claim."""
    with pytest.raises(ValidationError, match="never assessed"):
        PCOSProfileOutput(
            patient_id="p1",
            dominant_profile="androgenic_leaning",
            assignment_is_stable=True,
            indeterminate=False,
            androgenic_evidence_source="unavailable",
            rule_based_components_used=["pcos_adapter.prototype_similarity"],
        )


# -- 3. mixed requires at least two assessable domains ----------------------


def test_mixed_requires_at_least_two_assessable_domains() -> None:
    model = PrototypeSimilarityModel(min_observed_domains=1)

    one = model.predict(row(metabolic=1.0))
    assert "mixed" not in one.eligible_profiles
    assert "mixed" in one.ineligible_profiles

    two = model.predict(row(metabolic=1.0, reproductive=1.0))
    assert len([v for v in two.observed_domains]) >= MIXED_MIN_ASSESSABLE_DOMAINS
    assert "mixed" in two.eligible_profiles


# -- 4. unavailable domains are not zero-filled ----------------------------


def test_unavailable_domains_are_removed_not_zero_filled() -> None:
    """An unmeasured domain must be absent from the comparison, not set to 0.0.

    Zero is a meaningful value in z-score space -- it is the cohort mean -- so
    filling it in silently asserts that the patient was average on an axis
    nobody measured. Against a contrastive centroid carrying negative weights
    that assertion actively changes which profile wins.
    """
    model = PrototypeSimilarityModel()
    observed_only = row(metabolic=1.8, reproductive=0.9, symptom_burden=0.5)

    result = model.predict(observed_only)

    assert "biochemical_androgenic_evidence" not in result.observed_domains
    assert "biochemical_androgenic_evidence" in result.missing_domains
    # And the zero-filled version must NOT give the same answer, which is what
    # proves the absence is being honoured rather than quietly imputed.
    zero_filled = {k: (0.0 if v is None else v) for k, v in observed_only.items()}
    assert model.predict(zero_filled).similarities != result.similarities


def test_ineligible_profiles_are_absent_rather_than_scored_zero() -> None:
    """Removal, not suppression: no similarity value may survive for a dropped profile."""
    model = PrototypeSimilarityModel()
    result = model.predict(row(metabolic=1.8, reproductive=0.9, symptom_burden=0.5))

    for name in result.ineligible_profiles:
        assert name not in result.similarities
        assert name not in result.affinities
        assert name not in result.distances

    # Affinities over the surviving profiles must still be a distribution.
    assert sum(result.affinities.values()) == pytest.approx(1.0)


def test_the_output_schema_refuses_a_score_on_an_unassessable_domain() -> None:
    with pytest.raises(ValidationError, match="unassessable but carries a score"):
        PCOSProfileOutput(
            patient_id="p1",
            phenotype_domain_scores={"biochemical_androgenic_evidence": 0.0},
            domain_assessability={"biochemical_androgenic_evidence": False},
            rule_based_components_used=["pcos_adapter.prototype_similarity"],
        )


# -- 5. unstable patients have no dominant profile -------------------------


def test_unstable_assignment_yields_no_dominant_profile() -> None:
    model = PrototypeSimilarityModel()
    # A patient the similarity model IS confident about, so the only thing that
    # can withhold the label is the stability verdict.
    scores = row(metabolic=2.5, reproductive=0.2, symptom_burden=0.4, lh_amh_pattern=-1.0)
    assert model.predict(scores).dominant == "metabolic_leaning"

    unstable = summarize(model.predict(scores), is_stable=False)

    assert unstable["dominant_profile"] is None
    assert unstable["indeterminate"] is True
    assert any("stability checks" in reason for reason in unstable["indeterminate_reasons"])


def test_unassessed_stability_also_yields_no_dominant_profile() -> None:
    """"We did not check" is not evidence of stability."""
    model = PrototypeSimilarityModel()
    summary = summarize(model.predict(row(metabolic=1.8, reproductive=0.9, symptom_burden=0.5)))

    assert summary["dominant_profile"] is None
    assert summary["indeterminate"] is True
    assert summary["stability_assessed"] is False


def test_a_stable_assignment_may_publish_its_profile() -> None:
    """The gate must not be unconditional, or it would say nothing."""
    model = PrototypeSimilarityModel()
    # An unambiguously metabolic patient: high metabolic, low on the contrasting
    # LH/AMH axis, so the top two profiles are nowhere near a tie.
    result = model.predict(
        row(metabolic=2.5, reproductive=0.2, symptom_burden=0.4, lh_amh_pattern=-1.0)
    )
    assert result.dominant == "metabolic_leaning"
    assert not result.is_indeterminate

    summary = summarize(result, is_stable=True)
    assert summary["dominant_profile"] == "metabolic_leaning"
    assert summary["indeterminate"] is False


def test_the_output_schema_refuses_a_profile_without_a_stability_verdict() -> None:
    with pytest.raises(ValidationError, match="assignment_is_stable=True"):
        PCOSProfileOutput(
            patient_id="p1",
            dominant_profile="metabolic_leaning",
            indeterminate=False,
            assignment_is_stable=None,
            rule_based_components_used=["pcos_adapter.prototype_similarity"],
        )


def test_stability_engine_and_summarize_agree() -> None:
    """End to end: whatever the engine calls unstable must not reach the output."""
    model = PrototypeSimilarityModel()
    engine = PhenotypeStabilityEngine(n_bootstrap=25, seed=0)
    scores = row(metabolic=1.0, reproductive=0.95, ovarian=0.9, symptom_burden=0.9)

    report = engine.evaluate(scores, model)
    summary = summarize(model.predict(scores), is_stable=report.is_stable)

    if not report.is_stable:
        assert summary["dominant_profile"] is None
        assert summary["indeterminate"] is True


# -- 6. calibration never fits on held-out labels --------------------------


def test_calibrator_refuses_any_source_but_out_of_fold_training_predictions() -> None:
    rng = np.random.default_rng(0)
    prob = rng.uniform(size=120)
    y = (rng.uniform(size=120) < prob).astype(float)

    for forbidden in ("heldout", "test", "held_out_109", "", "validation"):
        with pytest.raises(ValueError, match="may only be fitted on"):
            PlattCalibrator().fit(y, prob, source=forbidden)

    assert PlattCalibrator().fit(y, prob, source=ALLOWED_CALIBRATION_FIT_SOURCE).is_fitted


def test_transform_never_refits_on_the_data_it_is_given() -> None:
    """Applying the calibrator to held-out data must not change its parameters."""
    rng = np.random.default_rng(1)
    train_prob = rng.uniform(size=200)
    y_train = (rng.uniform(size=200) < train_prob).astype(float)
    calibrator = PlattCalibrator().fit(y_train, train_prob, source=ALLOWED_CALIBRATION_FIT_SOURCE)
    before = (calibrator.coef_, calibrator.intercept_, calibrator.n_fit_)

    # A held-out set with a deliberately different distribution: if `transform`
    # adapted at all, these parameters would move.
    calibrator.transform(rng.uniform(size=109) * 0.2)

    assert (calibrator.coef_, calibrator.intercept_, calibrator.n_fit_) == before
    assert calibrator.fit_source_ == ALLOWED_CALIBRATION_FIT_SOURCE


def test_unfitted_calibrator_cannot_transform() -> None:
    with pytest.raises(RuntimeError, match="must be called before transform"):
        PlattCalibrator().transform([0.5])


# -- 7. five calibration bins of approximately equal count -----------------


def test_five_bins_carry_approximately_equal_counts() -> None:
    rng = np.random.default_rng(2)
    prob = rng.uniform(size=109)
    y = (rng.uniform(size=109) < prob).astype(float)

    bins = equal_frequency_bins(y, prob, n_bins=DEFAULT_N_BINS)

    assert DEFAULT_N_BINS == 5
    assert len(bins) == 5
    counts = [b["n"] for b in bins]
    assert sum(counts) == 109
    # np.array_split guarantees sizes differing by at most one.
    assert max(counts) - min(counts) <= 1


def test_each_bin_reports_a_count_and_a_binomial_interval() -> None:
    rng = np.random.default_rng(3)
    prob = rng.uniform(size=250)
    y = (rng.uniform(size=250) < prob).astype(float)

    for entry in equal_frequency_bins(y, prob):
        assert entry["n"] > 0
        assert 0.0 <= entry["observed_ci_lower"] <= entry["observed_rate"] <= 1.0
        assert entry["observed_rate"] <= entry["observed_ci_upper"] <= 1.0
        assert isinstance(entry["interpretable"], bool)


def test_sparse_bins_are_marked_uninterpretable() -> None:
    """A bin too thin to reason from must say so on the row itself."""
    rng = np.random.default_rng(4)
    prob = rng.uniform(size=25)
    y = (rng.uniform(size=25) < prob).astype(float)

    bins = equal_frequency_bins(y, prob)

    assert all(entry["n"] < 20 for entry in bins)
    assert not any(entry["interpretable"] for entry in bins)
