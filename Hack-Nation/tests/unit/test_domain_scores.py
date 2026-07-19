"""Domain composites: coverage math, weights, evidence qualification, thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.phenotype_domains import PhenotypeDomainScorer, load_domain_specs
from registry.loader import load_phenotype_domains
from tests.fixtures.synthetic_tabular import make_synthetic_cohort


@pytest.fixture
def registry() -> dict:
    return load_phenotype_domains()


@pytest.fixture
def fitted_scorer() -> PhenotypeDomainScorer:
    df = make_synthetic_cohort(n=200, seed=2, missing_rate=0.15)
    return PhenotypeDomainScorer().fit(df)


def test_specs_come_entirely_from_the_registry(registry):
    specs = load_domain_specs(registry)
    assert set(specs) == set(registry["domains"])
    biochemical = specs["biochemical_androgenic_evidence"]
    assert biochemical.evidence_source == "biochemical"
    assert biochemical.min_coverage_to_report == pytest.approx(0.25)
    shbg = next(f for f in biochemical.features if f.code == "shbg")
    assert shbg.direction == -1
    assert shbg.evidence_class == "biochemical"

    # The clinical half must stay free of assay weight: one biochemical feature
    # here would put absent-assay weight back in the symptom denominator, which
    # is what made observed cutaneous signs unassessable in the first place.
    clinical = specs["clinical_androgenic_evidence"]
    assert clinical.evidence_source == "symptoms"
    assert {f.evidence_class for f in clinical.features} == {"report"}
    assert {f.code for f in clinical.features} >= {"acne", "androgenic_alopecia", "hair_growth_face"}
    assert "skin_darkening" not in {f.code for f in clinical.features}


def test_coverage_is_the_observed_weight_fraction(fitted_scorer):
    """Coverage = sum(w_j * m_j) / sum(w_j) over the domain's declared features."""
    spec = fitted_scorer.specs["biochemical_androgenic_evidence"]
    # hirsutism belongs to the CLINICAL half and must not appear in this
    # domain's numerator, denominator, or observed list.
    observed = {"total_testosterone": 40.0, "hirsutism": 1.0}
    row = pd.DataFrame([observed])

    score = fitted_scorer.score_frame(row)["biochemical_androgenic_evidence"][0]

    in_domain = {f.code for f in spec.features} & set(observed)
    expected_weight = sum(f.weight for f in spec.features if f.code in in_domain)
    assert score.coverage == pytest.approx(expected_weight / spec.total_weight)
    assert sorted(score.observed_features) == sorted(in_domain)
    assert "hirsutism" not in score.observed_features
    assert "total_testosterone" not in score.missing_features
    assert "dheas" in score.missing_features


def test_score_is_the_weighted_mean_of_directed_z_scores(fitted_scorer):
    spec = fitted_scorer.specs["biochemical_androgenic_evidence"]
    stats = fitted_scorer.stats
    values = {"total_testosterone": 60.0, "shbg": 30.0}
    row = pd.DataFrame([values])

    score = fitted_scorer.score_frame(row)["biochemical_androgenic_evidence"][0]

    numerator, denominator = 0.0, 0.0
    for feature in spec.features:
        if feature.code not in values:
            continue
        z = (values[feature.code] - stats.means[feature.code]) / stats.stds[feature.code]
        numerator += feature.weight * feature.direction * z
        denominator += feature.weight

    assert score.score == pytest.approx(numerator / denominator)


def test_direction_flips_the_contribution(fitted_scorer):
    """shbg has direction -1: a high value must lower the androgenic score."""
    stats = fitted_scorer.stats
    # total_testosterone is pinned at its mean (z = 0) purely to clear the
    # domain's coverage threshold, so the sign comes from shbg alone.
    neutral = {"total_testosterone": stats.means["total_testosterone"]}
    high = pd.DataFrame([{**neutral, "shbg": stats.means["shbg"] + 2 * stats.stds["shbg"]}])
    low = pd.DataFrame([{**neutral, "shbg": stats.means["shbg"] - 2 * stats.stds["shbg"]}])

    high_score = fitted_scorer.score_frame(high)["biochemical_androgenic_evidence"][0]
    low_score = fitted_scorer.score_frame(low)["biochemical_androgenic_evidence"][0]

    assert high_score.score < 0 < low_score.score


def test_weights_are_honored(fitted_scorer):
    """total_testosterone (w=1.5) must move the score more than acne (w=0.5)."""
    stats = fitted_scorer.stats
    heavy = pd.DataFrame(
        [
            {
                "total_testosterone": stats.means["total_testosterone"]
                + stats.stds["total_testosterone"]
            }
        ]
    )
    light = pd.DataFrame([{"acne": stats.means["acne"] + stats.stds["acne"]}])

    heavy_score = fitted_scorer.score_frame(heavy)["biochemical_androgenic_evidence"][0]
    light_score = fitted_scorer.score_frame(light)["biochemical_androgenic_evidence"][0]

    assert heavy_score.coverage > light_score.coverage


def test_symptoms_alone_score_the_clinical_domain_and_not_the_biochemical_one(fitted_scorer):
    """Symptoms alone must never be exported as biochemical hyperandrogenism.

    The separation is now structural rather than a string qualifier: cutaneous
    signs land in their own domain, and the biochemical domain simply has
    nothing to say.
    """
    row = pd.DataFrame([{"hirsutism": 1.0, "acne": 1.0, "hair_growth_face": 1.0}])
    scored = fitted_scorer.score_frame(row)

    clinical = scored["clinical_androgenic_evidence"][0]
    assert clinical.score is not None
    assert clinical.evidence_source == "symptoms"

    biochemical = scored["biochemical_androgenic_evidence"][0]
    assert biochemical.score is None
    assert biochemical.is_assessable is False


def test_absent_assays_do_not_make_symptom_evidence_unassessable(fitted_scorer):
    """The bug this split exists to fix.

    Under the merged domain the four assay weights (4.5 of 8.0) sat in the
    denominator whether or not an assay was drawn, so three observed cutaneous
    signs scored 1.5/8.0 = 0.19 coverage and fell under the 0.25 floor. The
    patient was reported as having no androgenic evidence while three androgenic
    signs were recorded in front of us.
    """
    row = pd.DataFrame([{"acne": 1.0, "androgenic_alopecia": 1.0, "hair_growth_face": 1.0}])
    scored = fitted_scorer.score_frame(row)

    clinical = scored["clinical_androgenic_evidence"][0]
    assert clinical.is_assessable, "observed symptoms must be assessable without any assay"
    assert clinical.coverage == pytest.approx(1.5 / 3.5)
    assert scored["biochemical_androgenic_evidence"][0].is_assessable is False


def test_score_is_none_below_the_min_coverage_threshold(fitted_scorer):
    """reproductive requires coverage >= 0.34; one light feature is not enough."""
    spec = fitted_scorer.specs["reproductive"]
    row = pd.DataFrame([{"infertility_history": 1.0}])
    score = fitted_scorer.score_frame(row)["reproductive"][0]

    assert score.coverage < spec.min_coverage_to_report
    assert score.score is None
    assert score.is_reportable is False
    assert any("below the registry threshold" in w for w in score.warnings)


def test_score_is_reported_above_the_threshold(fitted_scorer):
    row = pd.DataFrame(
        [
            {
                "cycle_length": 40.0,
                "cycle_irregularity": 1.0,
                "menstrual_frequency_per_year": 6.0,
                "amenorrhea": 1.0,
            }
        ]
    )
    spec = fitted_scorer.specs["reproductive"]
    score = fitted_scorer.score_frame(row)["reproductive"][0]

    assert score.coverage >= spec.min_coverage_to_report
    assert score.score is not None
    assert score.is_reportable


def test_a_fully_unobserved_patient_gets_zero_coverage_and_no_score(fitted_scorer):
    row = pd.DataFrame([{code: np.nan for code in fitted_scorer.required_codes}])
    scores = fitted_scorer.score_frame(row)

    for name, values in scores.items():
        assert values[0].coverage == 0.0, name
        assert values[0].score is None, name


def test_scoring_requires_fitting_first():
    with pytest.raises(RuntimeError, match="fit()"):
        PhenotypeDomainScorer().score_frame(pd.DataFrame([{"acne": 1.0}]))


def test_reference_statistics_use_observed_values_only():
    df = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d"],
            "total_testosterone": [10.0, 20.0, 30.0, np.nan],
        }
    )
    scorer = PhenotypeDomainScorer().fit(df)
    assert scorer.stats.means["total_testosterone"] == pytest.approx(20.0)
    assert scorer.stats.n_observed["total_testosterone"] == 3


def test_manifest_records_weights_and_thresholds(fitted_scorer):
    manifest = fitted_scorer.manifest()
    assert "s_d = sum_j" in manifest["formula"]
    androgenic = manifest["domains"]["biochemical_androgenic_evidence"]
    assert androgenic["evidence_source"] == "biochemical"
    assert manifest["domains"]["clinical_androgenic_evidence"]["evidence_source"] == "symptoms"
    assert any(f["code"] == "shbg" and f["direction"] == -1 for f in androgenic["features"])
    assert manifest["reference_statistics"]["means"]


def test_score_matrix_matches_individual_scores(fitted_scorer):
    df = make_synthetic_cohort(n=25, seed=9)
    matrix = fitted_scorer.score_matrix(df)
    scored = fitted_scorer.score_frame(df)

    for name, scores in scored.items():
        for i, score in enumerate(scores):
            cell = matrix[f"domain_{name}"].iloc[i]
            if score.score is None:
                assert np.isnan(cell)
            else:
                assert cell == pytest.approx(score.score)
