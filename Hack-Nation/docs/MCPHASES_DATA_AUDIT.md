# mcPHASES Data Audit

**Date:** 2026-07-18 · **Method:** direct inspection of `datasets/mcphases/raw/`.

## Licence — read first

`LICENSE.txt` is the **PhysioNet Restricted Health Data License v1.5.0**
(© 2025 MIT Laboratory for Computational Physiology). This is a *restricted*
licence, not open data. Consequences that bind this work:

* raw data may not be redistributed;
* derived artifacts (processed tables, predictions) must be treated as
  restricted unless the licence is re-read and confirmed to permit otherwise;
* **model checkpoints trained on it are treated as non-redistributable** until
  that is explicitly checked. Recorded in the model card.

`ACCESS_REQUIRED.txt` sits alongside, confirming credentialed access.

## Cohort

| Item | Value |
|:--|--:|
| Participants | **42** (`subject-info.csv`, and 42 distinct ids in the hormone table — consistent) |
| Participant-days | **5,659** |
| Days per participant | min 38 · median 90 · max 210 |
| Study intervals | **2022** (3,698 days) and **2024** (1,961 days) |

### The single most important splitting constraint

**20 of 42 participants appear in BOTH the 2022 and 2024 intervals.**

A naive split on `(id, study_interval)` — or any split treating a participant's
two study periods as independent units — would place the same person's 2022 data
in training and their 2024 data in test. Their physiology, device, and reporting
habits are shared, so the model would be scored partly on someone it has already
seen. Grouping must be on **`id` alone**, carrying both intervals together.

This is precisely the leakage prompt_5 §5 warns about, and it is not
hypothetical here: it affects nearly half the cohort.

## Targets

### Cycle phase — usable, four classes, well balanced

| Phase | Days |
|:--|--:|
| Luteal | 1,912 |
| Follicular | 1,386 |
| Fertility | 1,281 |
| Menstrual | 1,079 |

One row has a null phase. Balance is good enough that macro-F1 is meaningful
without resampling.

### Hormones — two are well covered, one is not

| Hormone | Observed | Note |
|:--|--:|:--|
| LH | 5,339 / 5,659 (**94.3%**) | usable |
| Estrogen (E3G) | 5,338 / 5,659 (**94.3%**) | usable |
| **PdG** | 1,864 / 5,659 (**32.9%**) | sparse — see below |

PdG is observed on barely a third of days. It can still be a reconstruction
target, but the loss must be **masked to observed days only**; averaging over
unobserved days would train the model to predict the imputation rather than the
hormone, and would make the reported MAE a statement about the filler.

### Symptoms — ordinal, ~59% coverage, and dirty in places

Symptoms (`cramps`, `fatigue`, `moodswing`, `bloating`, `sorebreasts`, …) are
observed on ~58.8% of days as an ordinal scale:
`Not at all < Low < Moderate < High < Very High`.

**`headaches` is contaminated**: its observed values mix the ordinal labels with
bare numerals (`'2'`, `'3'`, `'4'`, `'5'`, `'High'`). It must be normalised or
excluded, not parsed naively — `pd.factorize` on it would silently invent an
ordering.

## Signals available

| File | Size | Content |
|:--|--:|:--|
| `heart_rate.csv` | 1.9 GB | intraday HR |
| `calories.csv` | 617 MB | intraday |
| `wrist_temperature.csv` | 303 MB | intraday |
| `distance.csv` / `steps.csv` | 233 / 217 MB | intraday activity |
| `estimated_oxygen_variation.csv` | 93 MB | intraday |
| `sleep.csv` | 53 MB | per-sleep-episode, with duration and latency |
| `glucose.csv` | 24 MB | **CGM**, timestamped |
| `heart_rate_variability_details.csv` | 24 MB | HRV |
| `hormones_and_selfreport.csv` | 680 KB | the targets |
| `resting_heart_rate.csv`, `sleep_score.csv`, `stress_score.csv` | small | daily summaries |

High-frequency files are timestamped and must be aggregated to participant-days
deterministically (mean / sd / min / max / day-night difference / missing
fraction), as prompt_5 §4 requires.

## Already-built consolidation

`participant_days.jsonl` exists (5,701 lines) and already carries the target
shape — one row per participant-day with a `values` dict and a parallel
`is_observed` mask:

```json
{"participant_id":"mcphases:1","study_day":1,"cycle_phase":"follicular",
 "values":{"lh":2.9,"e3g":94.2,"pdg":null,"resting_heart_rate":74.79,
           "wrist_temperature":34.20,"hrv_rmssd":null,"mean_glucose":99.07},
 "is_observed":{...}}
```

Produced by `scripts/consolidate_mcphases.py` / `prepare_mcphases.py`, which
`train_temporal.py:92-94` confirms has **no synthetic fallback** — it reads real
data or fails. So prompt_5 §4 is substantially done; it needs extending with the
wearable/CGM summaries not yet included, not rebuilding.

## Implications

1. **Group splits by `id` only.** Never by `(id, study_interval)` — 20
   participants span both.
2. **Mask the PdG loss to observed days.** 67% of its rows are absent.
3. **Clean or drop `headaches`.**
4. **42 participants is small.** A held-out test of ~8 participants gives wide
   intervals; report per-participant results, not only pooled day-level means.
5. **Baselines are mandatory before claiming the GRU helps** — the incumbent is
   an echo-state network (see `TEMPORAL_MODEL_AUDIT.md`), and it may already be
   competitive.
6. **Checkpoints are restricted-licence artifacts.**
