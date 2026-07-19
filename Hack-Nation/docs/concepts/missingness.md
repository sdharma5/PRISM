# Missingness

<div class="prism-safety">
<strong>Rule:</strong> a missing value never silently becomes zero, and never
silently becomes the column mean either. Missingness is information about the
care a person received, and it is modelled as such.
</div>

## Six kinds of absent

A blank cell in a clinical table can mean at least six different things, and
they have opposite implications. PRISM keeps them apart with
`MissingnessStatus`:

| Status | Meaning | Typical cause |
|:--|:--|:--|
| `observed` | A real value is present | — |
| `not_collected` | Nobody ordered or asked | Fasting insulin is rarely ordered |
| `not_available` | Collected, but PRISM cannot use it | Value outside the registry's plausible range |
| `not_applicable` | The question does not apply | Cycle length for a post-hysterectomy participant |
| `extraction_failed` | A model tried and could not | Unreadable table cell, inaudible audio |
| `intentionally_masked` | Hidden on purpose | Masked-autoencoder training, ablation testing |

`not_collected` and `not_applicable` are especially different. The first is
often *informative*: a clinician who did not order an androgen panel was
implicitly making a judgement, and that pattern correlates with the outcome.
The second carries no signal about severity at all. Collapsing both into `NaN`
throws that distinction away before modeling begins.

## The contract in code

`schemas/event.py` enforces the pairing:

```python
# An observed event must carry a value.
if self.missingness_status == "observed" and self.value is None:
    raise ValueError(...)

# A non-observed event must NOT carry one.
if self.missingness_status != "observed" and self.value is not None:
    raise ValueError(...)
```

There is no third state. An event cannot be quietly half-present.

## Downstream handling

**Snapshots.** `PatientSnapshot` carries a `missingness_mask` alongside
`values`, plus the list of events it excluded and why. Coverage is computable
for any expected variable set: `snapshot.coverage(expected_codes)`.

**Features.** `features/missingness.py` builds explicit indicator columns.
Imputation happens inside the training fold only — the imputer is fitted on
train and applied to test, never the reverse (see
`tests/unit/test_preprocessing_leakage.py`).

**Domain scores.** A composite score is an availability-weighted average, so it
degrades gracefully rather than pretending:

$$ s_d = \frac{\sum_{j \in d} w_j z_j m_j}{\sum_{j \in d} w_j m_j} $$

Every `DomainScore` reports its `coverage`, its `observed_features`, and its
`missing_features`. Below the domain's `min_coverage_to_report` threshold the
score is `None` — refusing to score is a valid output.

**Time series.** Every time-varying feature is carried as a triple —
`value`, `is_observed`, `time_since_last_observed` — never as a bare float. A
carried-forward hormone reading from nine days ago is not the same measurement
as one taken this morning, and the model is told which it is looking at.

**Imaging.** If pixel spacing is unknown, no physical measurement is emitted at
all. The quality gate returns `measurement_feasible=False` with reasons, and
the module abstains rather than reporting a volume in arbitrary units.

## Why this is worth the friction

The alternative — zero-fill or mean-impute at ingestion — produces a tidy matrix
in which absent evidence is indistinguishable from average evidence. A
participant with no androgen panel then looks like a participant with normal
androgens. The phenotype score, the cluster assignment, and the stability
analysis all inherit that error, and no downstream check can recover it, because
by then the information is genuinely gone.
