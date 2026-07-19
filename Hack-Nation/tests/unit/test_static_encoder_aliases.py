"""The frozen static artifact's mis-named feature slot must stay bridged.

``kottarathil-2020`` trained a slot called ``cycle_length`` on a column that is
actually menses duration (fitted scaler: mean 4.94, scale 1.42). The artifact
cannot be renamed without retraining, so
:data:`models.tabular.encoder.LEGACY_FEATURE_ALIASES` feeds that slot from
``menses_duration`` instead. These tests exist because the failure is silent:
routing a real ~30-day cycle length into a slot scaled for ~5 days inverts the
score rather than erroring.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.tabular.encoder import LEGACY_FEATURE_ALIASES, StaticClinicalEncoder

_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts/encoders/static_clinical"


@pytest.fixture(scope="module")
def encoder() -> StaticClinicalEncoder:
    if not (_ARTIFACT_DIR / "static_clinical_encoder.joblib").exists():
        pytest.skip("trained static encoder artifact is not present")
    loaded = StaticClinicalEncoder.load(_ARTIFACT_DIR)
    assert loaded.artifact is not None
    if loaded.artifact.source_dataset not in LEGACY_FEATURE_ALIASES:
        pytest.skip("artifact predates the aliased source dataset")
    return loaded


def test_menses_duration_drives_the_slot(encoder: StaticClinicalEncoder) -> None:
    """The aliased slot must respond to menses_duration.

    2.0 and 8.0 straddle the slot's training median of 5.0, so neither can be
    confused with the imputed value the way 5.0 itself would be.
    """
    short = encoder.predict_proba_from_features({"menses_duration": 2.0})
    long = encoder.predict_proba_from_features({"menses_duration": 8.0})
    assert short != pytest.approx(long)


def test_true_cycle_length_is_inert(encoder: StaticClinicalEncoder) -> None:
    """cycle_length is a variable this artifact never saw; it must not score.

    Every value must land on the imputed median. Pinning the whole range rather
    than one point is deliberate: a partial leak would otherwise show up only at
    the magnitudes a patient-facing form actually sends (~30-52).
    """
    baseline = encoder.predict_proba_from_features({})
    for value in (2.0, 5.0, 28.0, 45.0, 52.0):
        assert encoder.predict_proba_from_features({"cycle_length": value}) == pytest.approx(
            baseline
        ), f"cycle_length={value} moved the score; the slot is not properly bridged"


def test_true_cycle_length_never_displaces_menses_duration(
    encoder: StaticClinicalEncoder,
) -> None:
    """Supplying both must score identically to supplying menses_duration alone.

    This is the regression that matters: a patient-facing form sends a real
    cycle length (~30-52 days) alongside the bleeding duration. If the former
    wins the slot, the score inverts.
    """
    both = encoder.predict_proba_from_features({"menses_duration": 5.0, "cycle_length": 52.0})
    menses_only = encoder.predict_proba_from_features({"menses_duration": 5.0})
    assert both == pytest.approx(menses_only)


def test_batch_path_resolves_aliases_like_the_single_patient_path(
    encoder: StaticClinicalEncoder,
) -> None:
    """predict_proba and predict_proba_from_features must not disagree."""
    frame = pd.DataFrame([{"menses_duration": 5.0, "cycle_length": 52.0}])
    batch = float(encoder.predict_proba(frame)[0])
    single = encoder.predict_proba_from_features({"menses_duration": 5.0, "cycle_length": 52.0})
    assert batch == pytest.approx(single)


def test_alias_target_is_a_registered_variable() -> None:
    """Every alias must point at a code the registry actually defines."""
    import yaml

    registry_path = Path(__file__).resolve().parents[2] / "registry/variables.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    variables = registry["variables"]

    for dataset, aliases in LEGACY_FEATURE_ALIASES.items():
        for slot, canonical in aliases.items():
            assert canonical in variables, (
                f"{dataset}: alias {slot} -> {canonical} names no registry variable"
            )
