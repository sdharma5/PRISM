# Temporal Model Audit

**Date:** 2026-07-18 · **Method:** direct code and artifact inspection.

## Verdict

The temporal component is **not a stub, and not a trained sequence model
either.** It is an *echo-state network*: a randomly-initialised recurrent
encoder whose output feeds genuinely-trained linear heads, fitted on genuinely
real mcPHASES data with participant-grouped splits.

The existing metrics are **real numbers from real data** and should not be
discarded. What they are not is evidence that a *learned sequence encoder*
helps, because no sequence encoder was ever trained.

## What is real

| Component | Status | Evidence |
|:--|:--|:--|
| Ridge hormone head | **Trained** — closed-form | `models/temporal/heads.py:43-57` |
| Cycle-phase logistic head | **Trained** — gradient descent | `heads.py:156-161` |
| Symptom heads | **Trained** | `heads.py:221-226` |
| Participant-grouped splits | **Correct** | `metrics.json: split_strategy = grouped_kfold_by_participant` |
| Real data | **Yes** | 4,185 train / 1,516 val participant-days |
| GRU encoder | **NOT trained** — fixed random projections | `gru.py:267` |
| TCN encoder | **NOT trained** — fixed random filters | `tcn.py:30` |
| Persisted checkpoint | **None** | no `.pt`/`.joblib` under `artifacts/` |
| Loadable for inference | **No** | nothing to load; refit required every run |

There is **no `loss.backward()` or `optimizer.step()` anywhere** in
`models/temporal/`. The recurrent weights are never updated.

To be fair to the design, `gru.py:267` states this plainly and defends it:

> "a randomly initialised recurrent encoder followed by a *trained* linear head
> is an echo-state network, which is a legitimate and well-behaved model for
> short sequences and small cohorts — precisely this regime."

That argument is sound for 42 participants. The gap is not dishonesty; it is
that the architecture *claims* a GRU and the reported numbers cannot separate
"the GRU helps" from "random features plus a good linear head are enough."
Section 9 of prompt_5 asks for exactly that comparison, and it has never been run.

## A regression introduced by installing torch — must be fixed

`build_sequence_encoder(..., backend="auto")` picks `TorchGRU` when torch is
importable and `NumpyGRU` otherwise (`gru.py:363-372`). Until this session torch
was absent, so every run used `NumpyGRU`. **Installing torch for the ultrasound
work silently switched the temporal branch to `TorchGRU`.**

That matters because the two are not equivalent:

```
NumpyGRU reproducible across instances:  True    (takes seed=)
TorchGRU reproducible across instances:  False   (no seed parameter)
```

`TorchGRU.__init__` accepts no seed (`gru.py:310`) and builds a default-
initialised `nn.GRU`. So the temporal pipeline is now **non-deterministic across
processes**, and any metric regenerated today would not match the committed
artifact — for reasons unrelated to the data or the model.

The existing test suite does not catch this; the temporal tests still pass.

**Required fix (either is acceptable, first is preferred):**
1. Train the GRU for real, which is the task — and seed it.
2. Failing that, pin `backend: numpy` in the temporal config so the switch is a
   deliberate choice rather than a side effect of an unrelated install.

## Are the committed metrics reproducible?

**Not exactly, as of now** — see above. They were produced under `NumpyGRU` with
a fixed seed and are reproducible *in that configuration*. They are honest
numbers; they are simply no longer the numbers this environment produces.
`artifacts/experiments/exp_dynamic_state/metrics.json` records
`dataset_version: "unversioned"` and `git_commit: "unknown"`, which further
limits reproducibility.

## What is genuinely missing

1. A trained sequence encoder.
2. A persisted checkpoint and any `save`/`load`/`export_token` lifecycle.
3. Baselines (LOCF, participant mean, linear, random forest) to establish that
   the sequence model earns its complexity.
4. A held-out **test** split. The current protocol is grouped CV with a
   validation fold; there is no untouched final test set.
5. Reproducibility metadata (dataset version, commit).

## Conclusion

Do **not** delete the existing metrics — they are real. Do relabel the component:
it is an *echo-state temporal model*, not a trained GRU. The work now is to
train a real GRU, compare it honestly against both the echo-state baseline and
simpler baselines, and persist it.
