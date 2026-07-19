"""Smoke test: the temporal state model runs end to end on a tiny cohort.

Also asserts the property that matters most about the exported artifact: it
describes a **current state**, never a subtype or diagnosis.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation.temporal import (
    evaluate_temporal,
    format_ablation_table,
    missing_modality_ablation,
)
from models.temporal.gru import NumpyGRU, build_sequence_encoder
from models.temporal.state_model import (
    STATE_NOT_SUBTYPE_WARNING,
    TemporalStateModel,
    grouped_participant_split,
)
from models.temporal.tcn import NumpyTCN
from tests.fixtures.synthetic_cycles import generate_cohort

pytestmark = pytest.mark.smoke

#: Words that would misrepresent a state estimate as a trait or diagnosis.
FORBIDDEN_TRAIT_WORDS = ("subtype", "phenotype", "diagnosis", "diagnosed")


@pytest.fixture(scope="module")
def fitted():
    cohort = generate_cohort(n_participants=8, n_days=60, seed=0)
    groups = [d.participant_id for d in cohort.days]
    train_index, test_index = grouped_participant_split(groups, test_fraction=0.3, seed=0)
    train = [cohort.days[i] for i in train_index]
    test = [cohort.days[i] for i in test_index]
    model = TemporalStateModel(lookback_days=21, hidden_size=16, seed=0).fit(train)
    return model, test


def test_end_to_end_forward_pass(fitted, tmp_path):
    """Participant-days in, TemporalStateOutput and ModalityToken out."""
    model, test = fitted
    outputs = model.predict(test)
    assert outputs

    output = outputs[-1]
    assert output.state_embedding
    assert set(output.hormone_predictions) == {"lh", "e3g", "pdg"}
    assert set(output.cycle_phase_probabilities) == {
        "menstrual",
        "follicular",
        "peri_ovulatory",
        "luteal",
    }
    assert abs(sum(output.cycle_phase_probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= output.input_coverage <= 1.0
    assert output.lookback_days == 21

    token = model.to_token(output, source_dataset="synthetic_cycles")
    assert token.modality == "longitudinal_hormonal_state"
    path = token.write_json(tmp_path / "temporal_state_token.json")
    assert json.loads(path.read_text())["modality"] == "longitudinal_hormonal_state"


def test_output_is_state_never_a_subtype(fitted):
    """The state/trait boundary must be explicit in the artifact itself."""
    model, test = fitted
    output = model.predict(test)[0]
    token = model.to_token(output)

    assert "Not a subtype, diagnosis, or clinical decision." in output.interpretation
    assert STATE_NOT_SUBTYPE_WARNING in output.warnings
    assert STATE_NOT_SUBTYPE_WARNING in token.warnings

    # No structured feature may assert a trait.
    for key, value in token.structured_features.items():
        for word in FORBIDDEN_TRAIT_WORDS:
            assert word not in key.lower(), f"structured feature '{key}' implies a trait"
            if isinstance(value, str) and key != "interpretation":
                assert word not in value.lower()


def test_lookback_is_bounded_to_a_cycle():
    """The window is clamped to [14, 30] days."""
    cohort = generate_cohort(n_participants=3, n_days=50, seed=1)
    for requested, expected in ((5, 14), (21, 21), (99, 30)):
        model = TemporalStateModel(lookback_days=requested, hidden_size=8, seed=0)
        model.fit(cohort.days)
        assert model.spec is not None
        assert model.spec.lookback_days == expected


def test_numpy_encoders_work_without_torch():
    """Both encoder families have a functional torch-free path."""
    x = np.random.default_rng(0).normal(size=(4, 21, 12))
    gru = NumpyGRU(input_size=12, hidden_size=8, seed=0)
    assert gru(x).shape == (4, 8)
    tcn = NumpyTCN(input_size=12, hidden_size=8, seed=0)
    assert tcn(x).shape == (4, 8)
    assert tcn.receptive_field >= 8
    assert build_sequence_encoder(12, hidden_size=8, backend="numpy") is not None


def test_tcn_backend_runs_end_to_end():
    """The TCN is a real alternative, not dead code."""
    cohort = generate_cohort(n_participants=5, n_days=50, seed=2)
    model = TemporalStateModel(
        lookback_days=21, hidden_size=12, encoder_kind="tcn", backend="numpy", seed=0
    ).fit(cohort.days)
    outputs = model.predict(cohort.days)
    assert outputs
    assert len(outputs[0].state_embedding) == 12


def test_evaluation_and_ablation_run(fitted):
    """Metrics and the degradation table are produced without error."""
    model, test = fitted
    outputs = model.predict(test)
    metrics = evaluate_temporal(outputs, test)
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

    rows = missing_modality_ablation(test, model.predict)
    conditions = {r.condition for r in rows}
    assert {"full", "no_wearable", "no_cgm", "no_symptoms", "sparse_hormones"} <= conditions
    assert rows[0].condition == "full"
    assert isinstance(format_ablation_table(rows), str)

    # Removing a modality must not silently *improve* coverage.
    full = next(r for r in rows if r.condition == "full")
    no_wearable = next(r for r in rows if r.condition == "no_wearable")
    assert no_wearable.mean_input_coverage < full.mean_input_coverage


def test_model_card_forbids_subtype_use(fitted):
    """The card must rule out trait inference explicitly."""
    model, _ = fitted
    card = model.export_model_card_metadata()
    joined = " ".join(card.out_of_scope_uses).lower()
    assert "subtype" in joined
    assert "diagnosis" in joined
    assert "current" in card.intended_use.lower()
