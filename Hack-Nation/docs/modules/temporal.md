# Dynamic hormonal state (Step 9)

Learns a **current-state** representation from longitudinal data — not a subtype.
See [trait vs state](../concepts/trait_vs_state.md).

## Participant-day table

One row per participant per day: cycle day and phase, LH, E3G, PdG, menstrual
flow, daily symptoms, sleep, resting heart rate, HRV, temperature, activity, CGM
mean/variability/time-in-range — each with its missingness mask.

High-frequency wearable and CGM streams are aggregated to mean, median, standard
deviation, min, max, day–night difference, time in range, rate of change and
missing fraction. Every aggregation choice is documented in
`ingestion/mcphases/daily_aggregation.py`.

## Splits

Grouped by participant: leave-one-participant-out, grouped five-fold, or final
held-out participants.

**Days from one participant are never randomly split across train and test.**
Adjacent days are near-duplicates; splitting them produces a model that has
effectively memorized each participant and a metric that means nothing.
`tests/unit/test_temporal_splits.py` fails if this regresses.

## Architecture

```text
Previous 14–30 days → input projection → GRU → current-state embedding → heads
```

Inputs: observed daily values, missingness indicators, time since last
observation, cycle-day encoding, modality identifiers. GRU-D-style decay for
missing values is optional.

Every time-varying feature is carried as the triple
`(value, is_observed, time_since_last_observed)` — never a bare float. A
carried-forward reading from nine days ago is not this morning's measurement.

## Heads and loss

Hormone reconstruction (LH, E3G, PdG) · cycle state (menstrual, follicular,
peri-ovulatory, luteal) · next-day symptoms · masked reconstruction.

$$ \mathcal{L}_{state} = \lambda_h \mathcal{L}_{hormone} + \lambda_c \mathcal{L}_{cycle} + \lambda_s \mathcal{L}_{symptom} + \lambda_m \mathcal{L}_{masked} $$

Continuous: MSE, MAE or Gaussian NLL. Categorical: cross-entropy. Multilabel
symptoms: binary cross-entropy.

## Evaluation

**Hormones:** MAE · RMSE · Spearman · peak-timing error · interval coverage.
**Cycle state:** macro F1 · balanced accuracy · per-participant accuracy ·
calibration error. **Symptoms:** AUPRC · F1 · Brier.

**Robustness:** performance with no wearable data, no CGM, no symptoms, and
sparse hormone measurements — reported as a degradation table, because a model
that collapses without CGM is a CGM model wearing a hormone model's name.

## Output

`TemporalStateOutput` carries a state embedding, hormone predictions, cycle-phase
probabilities, symptom probabilities, uncertainty, input coverage — and a fixed
`interpretation` field stating it is a current-state estimate, not a subtype,
diagnosis, or clinical decision.
