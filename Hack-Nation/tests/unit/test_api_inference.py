"""API-level tests, including the invariants that must never regress.

The four invariants at the bottom of this file are the ones that matter
clinically. They are asserted against the real loaded encoders rather than
mocks, because each of them is a property of the *wiring* -- a mock static head
would happily return a score on the temporal-only route and the test would pass
while the service shipped the bug.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """One app, one model load, shared across this module's tests."""
    with TestClient(create_app()) as test_client:
        yield test_client


#: A patient presenting with irregular cycles and cutaneous androgenic signs,
#: and no androgen assay. Deliberately the symptoms-only case: it is the one the
#: language requirements care most about.
SYMPTOMATIC_FEATURES: dict[str, Any] = {
    "age": 29,
    "bmi": 31.4,
    "menses_duration": 6,
    "cycle_length": 48,
    "cycle_irregularity": True,
    "hirsutism": True,
    "acne": True,
    "androgenic_alopecia": True,
    "ferriman_gallwey_score": 9,
    "weight": 82,
    "height": 162,
    "luteinizing_hormone": 14.2,
    "follicle_stimulating_hormone": 5.1,
    "anti_mullerian_hormone": 7.8,
}


def _temporal_days(patient_id: str, n: int = 30) -> list[dict[str, Any]]:
    return [
        {
            "participant_id": patient_id,
            "study_day": i,
            "cycle_day": (i % 28) + 1,
            "values": {"urinary_lh": 5.0 + (i % 7), "e3g": 40.0 + i, "pdg": 3.0 + (i % 5)},
            "is_observed": {"urinary_lh": True, "e3g": True, "pdg": True},
            "time_since_last_observed": {"urinary_lh": 0.0, "e3g": 0.0, "pdg": 0.0},
        }
        for i in range(n)
    ]


# -- service surface -------------------------------------------------------


def test_health_does_not_require_models(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_model_status_reports_the_ultrasound_gate(client: TestClient) -> None:
    """The gated branch must be distinguishable from an absent one."""
    status = client.get("/api/v1/models/status").json()

    ultrasound = status["ovarian_ultrasound"]
    assert ultrasound["available"] is False
    assert ultrasound["validated_for_inference"] is False
    # Trained and persisted stay True: the checkpoint exists and loads. Saying
    # otherwise would describe a deliberate gate as a missing artifact.
    assert ultrasound["trained"] is True
    assert ultrasound["persisted"] is True
    assert ultrasound["reason"]

    assert status["static_clinical"]["available"] is True
    assert status["temporal_state"]["available"] is True


def test_static_inference_returns_a_scored_profile(client: TestClient) -> None:
    body = {"patient_id": "p-static", "clinical_features": SYMPTOMATIC_FEATURES}
    payload = client.post("/api/v1/patients/infer", json=body).json()

    assessment = payload["pcos_assessment"]
    assert assessment["available"] is True
    assert assessment["source"] == "static_clinical"
    assert 0.0 <= assessment["raw_model_score"] <= 1.0
    assert assessment["evidence_level"] != "not_available"
    assert assessment["qualifier"]

    assert payload["disclaimer"]
    assert "longitudinal_hormonal_state" in payload["missing_modalities"]


def test_ultrasound_inference_is_refused_with_its_reason(client: TestClient) -> None:
    response = client.post(
        "/api/v1/patients/infer/ultrasound", json={"patient_id": "p-us", "job_ids": []}
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["validated_for_inference"] is False
    assert "segmentation" in detail["reason"].lower()


def test_ultrasound_job_is_accepted_but_terminates_unavailable(client: TestClient) -> None:
    """Uploads may be built against; results may not be fabricated."""
    created = client.post("/api/v1/jobs/ultrasound", json={"patient_id": "p-us"}).json()
    assert created["status"] == "unavailable"
    assert created["reason"]
    assert created["result"] is None

    fetched = client.get(f"/api/v1/jobs/ultrasound/{created['job_id']}").json()
    assert fetched["job_id"] == created["job_id"]
    assert fetched["status"] == "unavailable"


def test_events_round_trip_and_unknown_patient_is_empty(client: TestClient) -> None:
    event = {
        "patient_id": "p-events",
        "variable_name": "cycle_length",
        "canonical_variable_code": "cycle_length",
        "value": 48.0,
        "modality": "questionnaire",
        "provenance": "patient_confirmed",
        "extraction_confidence": 0.9,
        "confirmation_status": "confirmed",
    }
    created = client.post("/api/v1/events", json={"events": [event]})
    assert created.status_code == 201
    assert created.json()["stored"] == 1

    stored = client.get("/api/v1/events/p-events").json()
    assert len(stored) == 1
    assert stored[0]["canonical_variable_code"] == "cycle_length"

    # Not yet onboarded is not an error.
    assert client.get("/api/v1/events/nobody").json() == []


def test_all_null_clinical_features_is_rejected(client: TestClient) -> None:
    """Placeholder nulls are a client bug, not an abstention."""
    response = client.post(
        "/api/v1/patients/infer",
        json={"patient_id": "p-null", "clinical_features": {"bmi": None, "age": None}},
    )
    assert response.status_code == 422


def test_mismatched_patient_id_is_a_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/v1/patients/infer",
        json={
            "patient_id": "p-a",
            "temporal_observations": _temporal_days("p-b", n=20),
        },
    )
    assert response.status_code == 422


def test_temporal_methods_are_translated(client: TestClient) -> None:
    """LOCF must never be described as a neural prediction."""
    body = {"patient_id": "p-temporal", "temporal_observations": _temporal_days("p-temporal")}
    state = client.post("/api/v1/patients/infer/temporal", json=body).json()["current_state"]

    assert state["available"] is True
    assert state["predicted_cycle_phase"]
    assert state["cycle_phase_probabilities"]

    # Keyed by canonical variable code, not a display abbreviation, so an
    # estimate joins to the events and registry entry for the same variable.
    lh = state["hormone_estimates"]["urinary_lh"]
    assert lh["code"] == "urinary_lh"
    assert lh["display_name"]
    assert lh["method_code"] == "locf"
    assert lh["method"] == "Based on the latest observed value"
    assert "predict" not in lh["method"].lower()


# -- required invariants ---------------------------------------------------


def test_invariant_temporal_alone_never_yields_a_pcos_score(client: TestClient) -> None:
    body = {"patient_id": "p-t-only", "temporal_observations": _temporal_days("p-t-only")}
    payload = client.post("/api/v1/patients/infer/temporal", json=body).json()

    assessment = payload["pcos_assessment"]
    assert assessment["available"] is False
    assert assessment["raw_model_score"] is None
    assert assessment["calibrated_model_score"] is None
    assert assessment["evidence_level"] == "not_available"
    assert assessment["unavailable_reason"]


def test_invariant_ultrasound_alone_never_yields_a_pcos_score(client: TestClient) -> None:
    """The route refuses outright, which is the strongest form of this guarantee."""
    response = client.post(
        "/api/v1/patients/infer/ultrasound", json={"patient_id": "p-u-only", "job_ids": ["j1"]}
    )
    assert response.status_code == 503
    assert "pcos_assessment" not in response.json()


def test_invariant_unstable_or_indeterminate_never_names_a_dominant_profile(
    client: TestClient,
) -> None:
    body = {"patient_id": "p-phen", "clinical_features": SYMPTOMATIC_FEATURES}
    phenotype = client.post("/api/v1/patients/infer", json=body).json()["phenotype"]

    if phenotype["indeterminate"] or not phenotype["stable_dominant_profile"]:
        assert phenotype["dominant_profile"] is None
        assert phenotype["status"] == "no_stable_dominant_profile"
        # Similarities are still shown: withholding a label is not the same as
        # withholding the evidence behind it.
        assert phenotype["profile_similarities"]


def test_invariant_symptoms_only_androgenic_result_carries_its_qualifier(
    client: TestClient,
) -> None:
    body = {"patient_id": "p-androgenic", "clinical_features": SYMPTOMATIC_FEATURES}
    payload = client.post("/api/v1/patients/infer", json=body).json()

    clinical = payload["rotterdam_axes"]["hyperandrogenism_clinical"]
    assert clinical["status"] == "met"
    # An axis describes its OWN kind of evidence. The report-level summary lives
    # at the top level, because the two can legitimately disagree: an axis can
    # fire on a guideline threshold while the matching domain stays unavailable
    # for want of training reference statistics.
    assert clinical["evidence_source"] == "clinical"
    # The absence of an assay must be stated on the axis itself, not left for
    # the frontend to infer from a missing sibling object.
    assert clinical["biochemical_evidence_available"] is False

    biochemical = payload["rotterdam_axes"]["hyperandrogenism_biochemical"]
    assert biochemical["status"] == "not_assessable"
    assert biochemical["evidence_source"] == "biochemical"

    # The symptoms-only qualifier itself, at the level it actually describes.
    assert payload["androgenic_evidence_source"] == "symptoms_only"


def test_stability_and_indeterminacy_do_not_contradict_each_other(client: TestClient) -> None:
    """A withheld profile must not be narrated as a confirmed stable one."""
    body = {"patient_id": "p-stab", "clinical_features": SYMPTOMATIC_FEATURES}
    phenotype = client.post("/api/v1/patients/infer", json=body).json()["phenotype"]

    if phenotype["indeterminate"]:
        plain = phenotype["stability"]["plain_language"]
        assert "No single profile is reported" in plain
        assert phenotype["stability"]["withheld_reason"]


# -- contract stability ----------------------------------------------------


def test_response_declares_its_schema_version_and_identity(client: TestClient) -> None:
    body = {"patient_id": "p-contract", "clinical_features": SYMPTOMATIC_FEATURES}
    payload = client.post("/api/v1/patients/infer", json=body).json()

    assert payload["schema_version"]
    assert payload["report_id"].startswith("rpt_")
    assert payload["generated_at"]
    # The framing is data, not a frontend constant, so no client can drop it.
    assert payload["is_diagnosis"] is False
    assert payload["disclaimer"]


def test_every_declared_field_is_fillable(client: TestClient) -> None:
    """No field may be structurally impossible to populate.

    A field the mapper can never fill is worse than an absent one: a client
    binds to it and renders null forever. This asserts that a maximally-supplied
    patient actually exercises the fields that are supposed to carry data --
    coverage in particular, which was declared but unreachable until the domain
    detail was plumbed through from the encoder.
    """
    body = {
        "patient_id": "p-full",
        "clinical_features": SYMPTOMATIC_FEATURES,
        "temporal_observations": _temporal_days("p-full"),
    }
    payload = client.post("/api/v1/patients/infer", json=body).json()

    assert payload["modality_coverage"] is not None
    assert payload["pcos_assessment"]["feature_coverage"] is not None

    scored = [d for d in payload["phenotype"]["domain_scores"].values() if d["available"]]
    assert scored, "expected at least one assessable domain"
    for domain in scored:
        assert domain["label"], "domain label must come from the registry"
        assert domain["coverage"] is not None
        assert domain["scale"] == "cohort_z_score"
        assert domain["observed_variables"]

    evidence = payload["supporting_evidence"]
    assert evidence
    assert any(item["variable_code"] for item in evidence)
    assert any(item["guideline_source"] for item in evidence)

    state = payload["current_state"]
    assert state["confidence"] is not None
    assert state["observed_days"] is not None

    records = payload["provenance"]["records"]
    assert records
    assert any(r["confidence"] is not None for r in records)
    assert any(r["origin"] == "rule_based_interpretation" for r in records)


def test_domain_display_order_is_total_and_stable(client: TestClient) -> None:
    """Dict order is not a contract; display_order is."""
    body = {"patient_id": "p-order", "clinical_features": SYMPTOMATIC_FEATURES}
    domains = client.post("/api/v1/patients/infer", json=body).json()["phenotype"]["domain_scores"]

    orders = [d["display_order"] for d in domains.values()]
    assert len(orders) == len(set(orders)), "display_order must be unique across domains"


def test_model_status_is_fully_typed(client: TestClient) -> None:
    """The frontend keys off this, so it must not be an open dict."""
    status = client.get("/api/v1/models/status").json()
    assert status["schema_version"]
    for branch in ("static_clinical", "temporal_state", "ovarian_ultrasound"):
        body = status[branch]
        assert set(body) <= {
            "available",
            "trained",
            "persisted",
            "validated_for_inference",
            "version",
            "implementation",
            "reason",
        }
    assert "available" in status["calibration"]


def test_axis_evidence_source_never_contradicts_axis_status(client: TestClient) -> None:
    """An axis that is met must not also report its evidence as unavailable.

    Regression for a real inconsistency: the report-level
    `androgenic_evidence_source` was stamped onto every androgenic axis, so a
    patient with a measured androgen produced `hyperandrogenism_biochemical:
    met` alongside `evidence_source: "unavailable"` -- two true facts about
    different things, rendered as one contradictory object.
    """
    body = {
        "patient_id": "p-assayed",
        "clinical_features": {
            **SYMPTOMATIC_FEATURES,
            "total_testosterone": 62.0,
            "dheas": 268.0,
            "shbg": 24.0,
        },
    }
    payload = client.post("/api/v1/patients/infer", json=body).json()

    for name, axis in payload["rotterdam_axes"].items():
        if axis["status"] == "met":
            assert axis["evidence_source"] != "unavailable", (
                f"axis {name} is met but reports its evidence as unavailable"
            )


def test_feature_coverage_counts_only_model_features(client: TestClient) -> None:
    """Coverage must not pool domain-composite misses with feature misses.

    Regression: `missing_fields` carries both kinds, and dividing the pooled
    count by the feature count made the ratio reach 1.0 for a patient who had
    supplied 5 of 19 features -- reporting "0% of the model's features were
    observed" to someone whose real coverage was 26%.
    """
    body = {"patient_id": "p-coverage", "clinical_features": SYMPTOMATIC_FEATURES}
    payload = client.post("/api/v1/patients/infer", json=body).json()

    coverage = payload["pcos_assessment"]["feature_coverage"]
    assert coverage is not None
    assert 0.0 < coverage <= 1.0, "a patient who supplied features cannot have 0 coverage"

    # The warning text and the reported number describe the same quantity and
    # must not contradict each other on screen.
    zero_warning = [w for w in payload["warnings"] if "0% of the model's features" in w]
    assert not zero_warning, f"coverage is {coverage} but a 0% warning was emitted"


# -- speech ----------------------------------------------------------------


def test_speech_availability_is_reported_not_assumed(client: TestClient) -> None:
    """The frontend disables the recorder from this, so it must always be present."""
    speech = client.get("/api/v1/models/status").json()["speech"]
    assert isinstance(speech["available"], bool)
    if not speech["available"]:
        # An unavailable capability must name what is missing, exactly as the
        # ultrasound gate does. "Unavailable" with no reason is unactionable.
        assert speech["reason"]


def test_empty_audio_upload_is_a_client_error(client: TestClient) -> None:
    """An empty upload is a 400, never a 500 or a fabricated transcript."""
    response = client.post(
        "/api/v1/speech/transcribe",
        files={"audio": ("empty.webm", b"", "audio/webm")},
        data={"patient_id": "p-speech"},
    )
    # 503 when the dependency is absent; 400 when it is present and the upload
    # is empty. Both are correct; a 200 with an invented transcript is not.
    assert response.status_code in {400, 503}
    assert "error" in response.json()["detail"]


# -- intake ----------------------------------------------------------------


def test_intake_schema_comes_from_the_registry(client: TestClient) -> None:
    """Labels, units and ranges must not be hardcoded in the frontend."""
    schema = client.get("/api/v1/intake/schema").json()

    assert schema["groups"]
    # A curated field naming a code the registry does not define would collect an
    # answer the encoder then discards, which looks to the patient like it was
    # recorded. Serving it at all is the bug.
    assert schema["dropped_unknown_codes"] == []

    fields = {f["code"]: f for g in schema["groups"] for f in g["fields"]}

    cycle = fields["cycle_length"]
    assert cycle["unit"] == "day"
    assert (cycle["min"], cycle["max"]) == (1.0, 365.0)

    menses = fields["menses_duration"]
    assert (menses["min"], menses["max"]) == (1.0, 14.0)
    # The two are trivially confusable and the confusion already cost this
    # project once, so each must carry text distinguishing it from the other.
    assert cycle["help_text"] and menses["help_text"]
    assert cycle["help_text"] != menses["help_text"]


def test_derivable_features_are_computed_not_imputed(client: TestClient) -> None:
    """Supplying weight and height must supply BMI.

    BMI is one of the model's 19 features and is derived during training. Not
    deriving it at inference meant a patient who gave weight and height had
    their BMI replaced by a training median -- discarding evidence they had
    actually provided, and skewing the score toward the cohort average.
    """
    with_parts = client.post(
        "/api/v1/patients/infer",
        json={
            "patient_id": "p-derive",
            "clinical_features": {"age": 29, "weight": 82.0, "height": 162.0},
        },
    ).json()

    without = client.post(
        "/api/v1/patients/infer",
        json={"patient_id": "p-derive", "clinical_features": {"age": 29}},
    ).json()

    assert (
        with_parts["pcos_assessment"]["feature_coverage"]
        > without["pcos_assessment"]["feature_coverage"]
    )
    # weight + height + the derived bmi, versus age alone.
    assert with_parts["pcos_assessment"]["raw_model_score"] != pytest.approx(
        without["pcos_assessment"]["raw_model_score"]
    )


def test_derivation_never_overwrites_a_supplied_value(client: TestClient) -> None:
    """A measured BMI must win over one derived from weight and height."""
    measured = client.post(
        "/api/v1/patients/infer",
        json={
            "patient_id": "p-bmi",
            "clinical_features": {"age": 29, "weight": 82.0, "height": 162.0, "bmi": 22.0},
        },
    ).json()
    derived = client.post(
        "/api/v1/patients/infer",
        json={
            "patient_id": "p-bmi",
            "clinical_features": {"age": 29, "weight": 82.0, "height": 162.0},
        },
    ).json()

    # 82kg at 1.62m derives to ~31.2, so a supplied 22.0 must produce a
    # different score -- otherwise the supplied value was silently discarded.
    assert measured["pcos_assessment"]["raw_model_score"] != pytest.approx(
        derived["pcos_assessment"]["raw_model_score"]
    )
