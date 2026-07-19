"""The committed demo fixtures must stay in step with the models.

Mock mode ships these files to the browser, so a stale fixture is a demo that
shows behaviour the system no longer has. `--check` in
``scripts/generate_demo_fixtures.py`` is the regeneration guard; the tests here
additionally assert that each fixture still exercises the state it exists for,
and that none of them violates an invariant the real API enforces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.api.demo_patients import DEMO_PATIENTS
from apps.api.schemas.responses import WebsitePCOSProfileResponse

_FIXTURE_DIR = Path(__file__).resolve().parents[2].parent / "UI/prism-app/src/lib/demo"


def _load(key: str) -> dict:
    path = _FIXTURE_DIR / f"{key}.json"
    if not path.exists():
        pytest.skip(f"demo fixtures not generated at {_FIXTURE_DIR}")
    return json.loads(path.read_text())


@pytest.mark.parametrize("patient", DEMO_PATIENTS, ids=lambda p: p.key)
def test_fixture_validates_against_the_response_contract(patient) -> None:
    """A fixture that no longer parses as the contract would break the client."""
    fixture = _load(patient.key)
    # extra="forbid" on the schema means an obsolete field fails here rather
    # than silently rendering as undefined in the browser.
    WebsitePCOSProfileResponse.model_validate(fixture["response"])


@pytest.mark.parametrize("patient", DEMO_PATIENTS, ids=lambda p: p.key)
def test_fixture_carries_its_framing(patient) -> None:
    response = _load(patient.key)["response"]
    assert response["is_diagnosis"] is False
    assert response["disclaimer"]
    assert response["schema_version"]


def test_sarah_is_the_symptoms_only_androgenic_case() -> None:
    """The profile the language requirements care most about."""
    response = _load("sarah")["response"]

    assert response["androgenic_evidence_source"] == "symptoms_only"
    assert response["rotterdam_axes"]["ovulatory_dysfunction"]["status"] == "met"
    assert response["rotterdam_axes"]["hyperandrogenism_clinical"]["status"] == "met"
    # No assay was drawn, so the biochemical axis must stay not assessable
    # rather than resolving to "not met", which would read as a negative result.
    assert response["rotterdam_axes"]["hyperandrogenism_biochemical"]["status"] == "not_assessable"
    assert response["rotterdam_axes"]["polycystic_ovarian_morphology"]["status"] == "not_assessable"


def test_control_exercises_the_low_evidence_path() -> None:
    response = _load("control")["response"]

    assert response["pcos_assessment"]["evidence_level"] == "low"
    assert response["rotterdam_axes"]["ovulatory_dysfunction"]["status"] == "not_met"


def test_maya_has_biochemical_evidence_unlike_sarah() -> None:
    """So the UI cannot assume the symptoms-only qualifier is always present."""
    response = _load("maya")["response"]

    biochemical = response["rotterdam_axes"]["hyperandrogenism_biochemical"]
    assert biochemical["status"] == "met"
    assert biochemical["evidence_source"] == "biochemical"
    # Maya supplies no longitudinal data, so this exercises the
    # "current state unavailable" layout while the static branch still reports.
    assert response["current_state"]["available"] is False
    assert response["current_state"]["unavailable_reason"]


@pytest.mark.parametrize("patient", DEMO_PATIENTS, ids=lambda p: p.key)
def test_fixture_never_names_an_unstable_dominant_profile(patient) -> None:
    """The demo must not show a label the real service would withhold."""
    phenotype = _load(patient.key)["response"]["phenotype"]

    if phenotype["indeterminate"] or not phenotype["stable_dominant_profile"]:
        assert phenotype["dominant_profile"] is None
        assert phenotype["status"] == "no_stable_dominant_profile"


def test_demo_index_lists_every_patient() -> None:
    index_path = _FIXTURE_DIR / "index.json"
    if not index_path.exists():
        pytest.skip("demo fixtures not generated")
    index = json.loads(index_path.read_text())
    assert {row["key"] for row in index} == {p.key for p in DEMO_PATIENTS}
