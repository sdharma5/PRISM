#!/usr/bin/env python
"""Tiny runnable demo of the dynamic hormonal-state model.

Fits on a small synthetic cohort with a participant-grouped split, prints one
participant's current-state estimate, shows the missing-modality degradation
table, and writes ``temporal_state_token.json`` next to this file.

The estimate is a **state**: where this person is today. It is not a subtype,
phenotype or diagnosis, and the token says so in its warnings.

Run:  python examples/temporal_example/run_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.temporal import (  # noqa: E402
    evaluate_temporal,
    format_ablation_table,
    missing_modality_ablation,
)
from models.temporal.state_model import TemporalStateModel, grouped_participant_split  # noqa: E402
from tests.fixtures.synthetic_cycles import generate_cohort  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    """Run the demo."""
    cohort = generate_cohort(n_participants=10, n_days=80, seed=0)
    groups = [day.participant_id for day in cohort.days]
    train_index, test_index = grouped_participant_split(groups, test_fraction=0.3, seed=0)
    train_days = [cohort.days[i] for i in train_index]
    test_days = [cohort.days[i] for i in test_index]

    train_ids = sorted({d.participant_id for d in train_days})
    test_ids = sorted({d.participant_id for d in test_days})
    print("=" * 72)
    print("Participant-grouped split (days from one person are NEVER split)")
    print("=" * 72)
    print(f"  train participants : {train_ids}")
    print(f"  test  participants : {test_ids}")
    print(f"  overlap            : {sorted(set(train_ids) & set(test_ids))}  <- must be empty")

    model = TemporalStateModel(lookback_days=21, hidden_size=32, seed=0).fit(train_days)
    assert model.report is not None
    print(
        f"\n  fitted on {model.report.n_windows} windows from "
        f"{model.report.n_participants} participants using {model.report.encoder}"
    )
    losses = {k: round(v, 4) for k, v in model.report.losses.items()}
    print(f"  training L_state components: {losses}")

    outputs = model.predict(test_days)
    print()
    print("=" * 72)
    print("One current-state estimate")
    print("=" * 72)
    output = outputs[-1]
    print(f"  participant        : {output.patient_id}")
    print(f"  as of              : {output.as_of_date}")
    print(f"  predicted phase    : {output.predicted_phase()}")
    print(
        "  phase probabilities: "
        + ", ".join(f"{k}={v:.2f}" for k, v in output.cycle_phase_probabilities.items())
    )
    print(
        "  hormones           : "
        + ", ".join(f"{k}={v:.1f}" for k, v in output.hormone_predictions.items())
    )
    print(
        "  next-day symptoms  : "
        + ", ".join(f"{k}={v:.2f}" for k, v in output.symptom_probabilities.items())
    )
    print(f"  input coverage     : {output.input_coverage:.2f}")
    print(f"  interpretation     : {output.interpretation}")

    metrics = evaluate_temporal(outputs, test_days)
    print("\n  test metrics:")
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "calibration_error", "macro_auprc"):
        if key in metrics:
            print(f"    {key:<20} {metrics[key]:.4f}")

    print()
    print("=" * 72)
    print("Missing-modality degradation")
    print("=" * 72)
    rows = missing_modality_ablation(test_days, model.predict)
    print(format_ablation_table(rows))

    token = model.to_token(output, source_dataset="synthetic_cycles")
    path = token.write_json(HERE / "temporal_state_token.json")
    print(f"\n  token written to : {path}")
    print(f"  token modality   : {token.modality}")
    print("  token warnings   :")
    for warning in token.warnings:
        print(f"    - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
