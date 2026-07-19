# Trait vs state

<div class="prism-safety">
PRISM models two different things and refuses to conflate them: what is
<strong>stable</strong> about a person's endocrine phenotype, and what is
<strong>true today</strong>.
</div>

## The distinction

| | Trait | State |
|:--|:--|:--|
| Question | "What is this person's stable phenotype?" | "Where is this person right now?" |
| Timescale | Months to years | Hours to days |
| Data | Cross-sectional clinical + labs + morphology | Longitudinal hormones, wearables, CGM, symptoms |
| Module | Steps 3–5 (static, domains, profiles) | Step 9 (temporal GRU) |
| Output | `PhenotypeProfile`, `StabilityReport` | `TemporalStateOutput` |
| Dataset | Public PCOS cohort | mcPHASES |

## Why the separation is load-bearing

A single LH measurement is close to uninterpretable without knowing the cycle
day it was drawn on. An LH of 12 mIU/mL is unremarkable mid-surge and notable in
the early follicular phase. Treat that number as a trait and you have encoded
sampling time as pathology.

Run it the other way and the error is just as bad: a luteal-phase progesterone
rise is a *state* observation, and reading it as evidence about a person's
long-run ovulatory phenotype ignores that a single ovulatory cycle says little
about the distribution of cycles.

There is also a hard data constraint. The two live in different datasets
describing different people (see
[ADR-002](../decisions/ADR-002-no-fake-pairing.md)), so a model that mixed them
would be mixing populations as well as timescales.

## What the temporal model does *not* output

`TemporalStateOutput` carries a fixed, non-optional field:

```python
interpretation: str = (
    "Current hormonal-state estimate. Not a subtype, diagnosis, or clinical decision."
)
```

It travels with every serialized token. The temporal module predicts current
cycle phase, reconstructs hormone levels, and forecasts next-day symptoms. It
does **not** emit a subtype, a diagnosis, or a risk score, and it is trained on
a dataset whose registry entry explicitly prohibits `pcos_subtype_validation`
and `pcos_diagnosis` — so an attempt to use it that way fails at
`DatasetRegistry.require()`, not in review.

## Where they would eventually meet

A person's trait phenotype should inform the prior over their state
trajectories, and their observed state trajectory is evidence about their trait.
That coupling is real, and it is exactly what a fusion model would learn.

It is not implemented, and cannot honestly be, until the same people have both
kinds of data. Until then the two modules stay independent, export separate
tokens under a shared envelope, and are validated separately on their own
held-out populations.
