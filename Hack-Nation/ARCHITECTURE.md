# Architecture

## The shape of the problem

Hormonal health is measured badly. The evidence about one person arrives from
incompatible channels — a questionnaire, a lab panel drawn on an unknown cycle
day, a sentence spoken in a consultation, a PDF from another clinic, an
ultrasound read by one sonographer — at different times, in different units,
with different reliability, and frequently in contradiction.

Most pipelines flatten that into a wide table early, because models want
matrices. PRISM flattens it as **late as possible**, and makes the flattening a
parameterized, reproducible, inspectable operation rather than an ingestion
side effect.

## Five layers

```text
   ingestion  →  event store  →  features  →  models  →  tokens
   (adapters)    (append-only)   (snapshots)  (per-modality)  (shared envelope)
```

### 1. Registry and ingestion

`registry/` is the authority. `datasets.yaml` declares what each dataset
contains, what it may support, and what it may never claim; `variables.yaml`
defines ~65 canonical variables with types, units and plausible ranges;
`units.yaml` holds every conversion factor (each with a unit test);
`phenotype_domains.yaml` defines domain composition.

Every adapter implements the same four-method contract:

```python
class BaseIngestionAdapter(ABC):
    def load_raw(self) -> object: ...        # never alters source files
    def validate_raw(self, raw) -> None: ...  # loud, descriptive failure
    def transform(self, raw) -> Iterable[HormonalHealthEvent]: ...
    def build_manifest(self) -> dict: ...     # version, checksums, mappings
```

An adapter calls `DatasetRegistry.require(dataset_id, use)` and **fails closed**
on a use the registry prohibits. Asking NHANES for `temporal_state_model` raises
`PermissionError` — the prohibition is executable, not advisory.

Invalid values are *reported*, never silently corrected: an out-of-range
potassium becomes a `not_available` event with the reason logged to
`dropped_records.csv`, not a clipped number.

### 2. The event store

One append-only store of `HormonalHealthEvent` records. Nothing is mutated;
nothing is deleted; conflicts are preserved and labelled rather than resolved.
Model-ready matrices come from `build_snapshot(...)`, which takes an `as_of`
time, a set of allowed confirmation statuses, and a modality filter, and returns
the selected values *plus* the missingness mask, the conflicts, and the list of
what it excluded and why.

Rationale: [ADR-001](docs/decisions/ADR-001-event-store.md).

### 3. Features

`features/` turns snapshots into matrices with explicit missingness indicators
and derived quantities (LH/FSH ratio, HOMA-IR, waist-hip ratio) computed only
when their inputs are genuinely observed. Every run writes a feature manifest
recording exactly which columns and transforms produced the matrix.

Transparent domain scores live here too — deterministic, registry-driven,
coverage-reporting, and readable by a clinician who wants to argue with them.

### 4. Models

Seven independent module families, each with its own config, loader,
preprocessor, model, training script, evaluation script, output contract, tests
and docs:

| Family | Produces |
|:--|:--|
| `models/tabular/` | Static baselines + masked autoencoder embedding |
| `models/phenotype/` | Domain scores, clustering, prototype mapping |
| `models/stability/` | Bootstrap, ablation, perturbation, abstention |
| `models/speech/` | Symptom token from confirmed events |
| `models/documents/` | Document token from grounded lab events |
| `models/ultrasound/` | Segmentation, instances, morphology |
| `models/temporal/` | GRU current-state representation |
| `models/adapters/pcos/` | **The only PCOS-specific code in the repository** |

The adapter boundary is the condition-agnostic guarantee. The event store, the
encoders and the token envelope know about *variables*; they do not know what
PCOS is. Supporting a second hormonal condition means new registry entries and a
new adapter — not a reshaped substrate.

### 5. Tokens

Each encoder exports a `ModalityToken` under one envelope: `patient_id`,
`modality`, `embedding`, `structured_features`, `quality_score`,
`confidence_score`, `observed_at`, `model_version`, `source_dataset`,
`provenance_ids`, `missing_fields`, `warnings`.

Five tokens exist at the end of Step 9: **static**, **symptom**, **document**,
**ultrasound**, **temporal state**.

They share an envelope so they are *comparable*. They are **not** concatenated,
because they describe different people —
[ADR-002](docs/decisions/ADR-002-no-fake-pairing.md). No fusion model is trained
in this phase, and a synthetic combined case may illustrate a UI but may never
appear in an evaluation.

## Cross-cutting invariants

These hold everywhere and each has a regression test that fails loudly:

1. **No preprocessing leakage.** Imputers, scalers and encoders are fitted
   inside the training fold only.
2. **Patient-level splits.** Longitudinal splits are grouped by participant;
   days from one person never straddle train and test.
3. **Missing is never zero.** Six distinct missingness states, enforced at the
   schema level.
4. **Unconfirmed evidence never reaches a model.** `is_model_ready` gates
   snapshot inclusion.
5. **Model-generated ≠ measured.** Imaging output carries
   `clinician_review_status` until a clinician changes it.
6. **Hedged language, enforced in code.** Banned-phrase guard on generated
   profile descriptions.
7. **Abstention is a valid output.** Every phenotype and imaging path can
   decline to answer, with reasons.

## Optional dependencies

torch, xgboost, pydicom, scikit-image, whisper and pdfplumber are all optional.
Each has a documented fallback — a threshold segmenter for torch-free imaging, a
numpy GRU, sklearn gradient boosting, a text-fixture document parser, a scripted
transcription adapter. CI runs the full suite **without** the extras in a
dedicated job, so the fallbacks stay real rather than aspirational.

## Where the architecture deliberately stops

No clinician directory, insurance filter, phone script, doctor-facing PDF,
treatment recommendation, or care-navigation layer. Those are downstream of a
validated core, and the core is not validated yet.
