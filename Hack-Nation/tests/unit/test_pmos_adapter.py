"""The PMOS adapter: not_assessable on missing inputs, and a hedged end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest

from models.adapters.pmos.adapter import PmosAdapter, PmosAdapterConfig
from models.adapters.pmos.diagnostic_features import FEATURE_AXES, assess_all_axes, assess_axis
from models.adapters.pmos.output_schema import PmosResearchOutput
from models.adapters.pmos.phenotype_heads import build_phenotype_profile, compute_domain_scores
from models.phenotype.clustering import ClusteringInput
from models.phenotype.prototype_mapping import assert_hedged_language
from schemas.phenotype import INDETERMINATE
from tests.fixtures.synthetic_clusters import make_synthetic_cluster_frame

# ------------------------------------------------- not_assessable on missing


def test_every_axis_is_not_assessable_with_no_inputs():
    axes = assess_all_axes({})
    assert set(axes) == set(FEATURE_AXES)
    for axis in axes.values():
        assert axis.status == "not_assessable"
        assert axis.evidence_available is False
        assert axis.is_met is False
        assert any("must never be read as absent" in w for w in axis.warnings)


def test_explicit_none_is_not_assessable_not_not_met():
    axes = assess_all_axes({"total_testosterone": None, "free_testosterone": None, "dheas": None})
    assert axes["hyperandrogenism_biochemical"].status == "not_assessable"


@pytest.mark.parametrize("axis_name", list(FEATURE_AXES))
def test_each_axis_individually_returns_not_assessable_when_its_inputs_are_missing(axis_name):
    spec = FEATURE_AXES[axis_name]
    assessment = assess_axis(spec, dict.fromkeys(spec.codes))
    assert assessment.status == "not_assessable"
    assert assessment.missing_codes


def test_axes_can_be_met_when_inputs_cross_documented_thresholds():
    axes = assess_all_axes(
        {
            "cycle_length": 45.0,
            "total_testosterone": 3.4,
            "ferriman_gallwey_score": 8.0,
            "follicle_number_per_ovary": 24.0,
        }
    )
    assert axes["ovulatory_dysfunction"].status == "met"
    assert axes["hyperandrogenism_biochemical"].status == "met"
    assert axes["hyperandrogenism_clinical"].status == "met"
    assert axes["polycystic_ovarian_morphology"].status == "met"


def test_axis_is_not_met_when_measured_below_threshold_but_records_gaps():
    axes = assess_all_axes({"cycle_length": 28.0})
    ovulatory = axes["ovulatory_dysfunction"]
    assert ovulatory.status == "not_met"
    assert ovulatory.evidence_available is True
    assert "menstrual_frequency_per_year" in ovulatory.missing_codes
    assert any("partial input set" in w for w in ovulatory.warnings)


def test_every_applied_threshold_carries_a_source():
    axes = assess_all_axes({"cycle_length": 45.0, "total_testosterone": 3.4})
    for axis in axes.values():
        for key, source in axis.threshold_sources.items():
            assert source.strip(), key


def test_assay_dependent_axes_are_flagged():
    axes = assess_all_axes({"total_testosterone": 3.4})
    assert axes["hyperandrogenism_biochemical"].assay_dependent
    assert any("placeholders" in w for w in axes["hyperandrogenism_biochemical"].warnings)


def test_amh_has_no_hard_coded_pcom_threshold():
    rule = next(
        r
        for r in FEATURE_AXES["polycystic_ovarian_morphology"].rules
        if r.code == "anti_mullerian_hormone"
    )
    assert rule.threshold is None
    assert (
        assess_all_axes({"anti_mullerian_hormone": 90.0})["polycystic_ovarian_morphology"].status
        == "not_assessable"
    )


def test_threshold_overrides_are_respected():
    values = {"total_testosterone": 2.0}
    assert assess_all_axes(values)["hyperandrogenism_biochemical"].status == "not_met"
    overridden = assess_all_axes(
        values, {"hyperandrogenism_biochemical": {"total_testosterone": 1.5}}
    )
    assert overridden["hyperandrogenism_biochemical"].status == "met"


# ------------------------------------------------------------- domain scores


def test_domain_score_is_withheld_below_registry_coverage():
    scores = compute_domain_scores({"bmi": 1.0})
    assert scores["metabolic"].score is None
    assert scores["metabolic"].coverage < 0.34
    assert scores["metabolic"].is_reportable is False


def test_domain_score_direction_is_applied():
    high = compute_domain_scores(
        {
            "bmi": 2.0,
            "fasting_insulin": 2.0,
            "homa_ir": 2.0,
            "hdl_cholesterol": -2.0,
            "waist_circumference": 2.0,
        }
    )["metabolic"]
    assert high.score is not None and high.score > 0


def test_symptom_only_androgenic_evidence_scores_the_clinical_domain_alone():
    """Cutaneous signs support the clinical domain and say nothing biochemical."""
    scores = compute_domain_scores(
        {
            "hirsutism": 1.0,
            "acne": 1.0,
            "androgenic_alopecia": 1.0,
            "ferriman_gallwey_score": 1.5,
        }
    )
    assert scores["clinical_androgenic_evidence"].score is not None
    assert scores["clinical_androgenic_evidence"].evidence_source == "symptoms"
    assert scores["biochemical_androgenic_evidence"].score is None


def test_assay_only_androgenic_evidence_scores_the_biochemical_domain_alone():
    scores = compute_domain_scores(
        {"total_testosterone": 2.0, "shbg": -1.0, "free_testosterone": 1.0}
    )
    assert scores["biochemical_androgenic_evidence"].score is not None
    assert scores["biochemical_androgenic_evidence"].evidence_source == "biochemical"
    assert scores["clinical_androgenic_evidence"].score is None


def test_build_profile_reserves_indeterminate_mass():
    profile = build_phenotype_profile(
        "p1", {}, {"profile_0": 0.8, "profile_1": 0.2}, indeterminate_mass=1.0
    )
    assert profile.dominant_profile == INDETERMINATE
    assert profile.phenotype_probabilities[INDETERMINATE] == 1.0


# --------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def fitted_adapter():
    frame, _ = make_synthetic_cluster_frame(n_per_group=25, seed=0)
    representation = ClusteringInput(
        label="raw_standardized",
        matrix=frame.to_numpy(),
        participant_ids=list(frame.index),
        feature_names=list(frame.columns),
    )
    config = PmosAdapterConfig(
        algorithms=("kmeans", "agglomerative"),
        k_values=(2, 3, 4),
        seeds=(0, 1),
        n_bootstrap=8,
        consensus_resamples=8,
        n_noise_replicates=2,
    )
    adapter = PmosAdapter(config).fit([representation], list(frame.index))
    return adapter, frame


def test_adapter_discovers_the_planted_number_of_groups(fitted_adapter):
    adapter, _ = fitted_adapter
    assert adapter.discovery is not None
    assert adapter.discovery.selection.k == 3


def test_profile_returns_the_documented_triple_with_hedged_language(fitted_adapter):
    adapter, frame = fitted_adapter
    patient_id = str(frame.index[0])
    raw = {"cycle_length": 45.0, "total_testosterone": 3.1}
    standardized = {c: float(frame.loc[patient_id, c]) for c in frame.columns}

    phenotype, stability, output = adapter.profile(patient_id, raw, standardized)

    assert phenotype.patient_id == patient_id == stability.patient_id == output.patient_id
    assert isinstance(output, PmosResearchOutput)
    assert sum(phenotype.phenotype_probabilities.values()) == pytest.approx(1.0)
    assert output.non_diagnostic_statement
    assert "does not diagnose" in output.non_diagnostic_statement
    assert output.limitations
    assert_hedged_language(output.model_organized_phenotype.resemblance_statement or "")
    for description in output.model_organized_phenotype.profile_descriptions.values():
        assert_hedged_language(description)


def test_output_separates_observed_from_missing_evidence(fitted_adapter):
    adapter, frame = fitted_adapter
    patient_id = str(frame.index[0])
    _, _, output = adapter.profile(patient_id, {"cycle_length": 45.0})

    assert output.observed_evidence.axis_status["ovulatory_dysfunction"] == "met"
    assert "polycystic_ovarian_morphology" in output.missing_evidence.not_assessable_axes
    assert (
        output.observed_evidence.axis_evidence_available["polycystic_ovarian_morphology"] is False
    )
    assert output.missing_evidence.notes


def test_uncertainty_and_k_rationale_are_always_populated(fitted_adapter):
    adapter, frame = fitted_adapter
    _, stability, output = adapter.profile(str(frame.index[0]))
    assert output.uncertainty.calibration_temperature is not None
    assert output.uncertainty.cohort_mean_bootstrap_jaccard is not None
    assert output.uncertainty.highest_fragility_feature in list(frame.columns)
    assert "No prior favoured any particular K" in (
        output.model_organized_phenotype.k_selection_rationale or ""
    )
    assert 0.0 <= stability.stability_score <= 1.0


def test_clear_cut_participant_is_not_abstained_on(fitted_adapter):
    adapter, frame = fitted_adapter
    standardized = {c: float(frame.iloc[0][c]) for c in frame.columns}
    _, stability, output = adapter.profile(str(frame.index[0]), None, standardized)
    assert stability.abstain is False
    assert output.abstention.abstained is False
    assert output.model_organized_phenotype.dominant_probability is not None


def test_abstention_moves_all_mass_to_indeterminate():
    """A participant with almost no observed evidence must not be named."""
    frame, _ = make_synthetic_cluster_frame(n_per_group=25, seed=1)
    representation = ClusteringInput(
        label="raw_standardized",
        matrix=frame.to_numpy(),
        participant_ids=list(frame.index),
        feature_names=list(frame.columns),
    )
    config = PmosAdapterConfig(
        algorithms=("kmeans",),
        k_values=(3,),
        seeds=(0,),
        n_bootstrap=5,
        consensus_resamples=5,
        n_noise_replicates=1,
    )
    adapter = PmosAdapter(config).fit([representation], list(frame.index))
    patient_id = str(frame.index[0])
    sparse = {c: None for c in frame.columns}
    sparse[str(frame.columns[0])] = 0.4

    phenotype, stability, output = adapter.profile(patient_id, {}, sparse)
    assert stability.abstain
    assert any("insufficient_observed_evidence" in r for r in stability.abstain_reasons)
    assert phenotype.phenotype_probabilities[INDETERMINATE] == 1.0
    assert phenotype.dominant_profile == INDETERMINATE
    assert output.abstention.indeterminate_probability == 1.0
    assert "indeterminate" in (output.model_organized_phenotype.resemblance_statement or "").lower()


def test_profile_before_fit_and_out_of_cohort_fail_loudly(fitted_adapter):
    adapter, _ = fitted_adapter
    with pytest.raises(RuntimeError, match="before fit"):
        PmosAdapter().profile("nobody")
    with pytest.raises(KeyError, match="not in the fitted cohort"):
        adapter.profile("not_a_participant")


def test_fit_refuses_an_empty_training_subset():
    frame, _ = make_synthetic_cluster_frame(n_per_group=10, seed=0)
    representation = ClusteringInput(
        label="raw_standardized",
        matrix=frame.to_numpy(),
        participant_ids=list(frame.index),
        feature_names=list(frame.columns),
    )
    with pytest.raises(ValueError, match="cluster_subset_ids is required"):
        PmosAdapter().fit([representation], [])


def test_output_round_trips_through_json(fitted_adapter, tmp_path):
    adapter, frame = fitted_adapter
    _, _, output = adapter.profile(str(frame.index[0]))
    path = output.write_json(tmp_path / "out.json")
    assert PmosResearchOutput.read_json(path).patient_id == output.patient_id


def test_non_diagnostic_statement_cannot_be_blanked(fitted_adapter):
    adapter, frame = fitted_adapter
    _, _, output = adapter.profile(str(frame.index[0]))
    payload = output.model_dump()
    payload["non_diagnostic_statement"] = "  "
    with pytest.raises(ValueError, match="mandatory"):
        PmosResearchOutput.model_validate(payload)


def test_clustering_only_used_the_named_subset():
    """The adapter must cluster the supplied subset, not the whole frame."""
    frame, _ = make_synthetic_cluster_frame(n_per_group=25, seed=0)
    representation = ClusteringInput(
        label="raw_standardized",
        matrix=frame.to_numpy(),
        participant_ids=list(frame.index),
        feature_names=list(frame.columns),
    )
    subset = list(frame.index[:40])
    config = PmosAdapterConfig(
        algorithms=("kmeans",),
        k_values=(2, 3),
        seeds=(0,),
        n_bootstrap=5,
        consensus_resamples=5,
        n_noise_replicates=1,
    )
    adapter = PmosAdapter(config).fit([representation], subset)
    assert adapter.discovery is not None
    assert adapter.discovery.data.participant_ids == subset
    assert len(adapter.discovery.fitted.labels) == len(subset)
    assert np.all(np.isin(adapter.discovery.fitted.labels, [0, 1, 2]))
