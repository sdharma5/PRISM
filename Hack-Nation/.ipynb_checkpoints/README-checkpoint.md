<h1 align="center">PRISM</h1>
<p align="center"><b>Personalized Reproductive and Integrated Systemic Model</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research_preview-ff1493?style=flat-square" alt="status: research preview">
  <img src="https://img.shields.io/badge/scope-steps_1--9-ff1493?style=flat-square" alt="scope: steps 1-9">
  <img src="https://img.shields.io/badge/not_for-clinical_use-ff1493?style=flat-square" alt="not for clinical use">
  <img src="https://img.shields.io/badge/python-3.11%2B-ff1493?style=flat-square" alt="python 3.11+">
  <img src="https://img.shields.io/badge/license-Apache_2.0-ff1493?style=flat-square" alt="Apache 2.0">
</p>

---

PRISM is a modular, condition-agnostic hormonal-health pipeline. It converts several
different forms of patient information — tabular clinical records, spoken symptom
narration, laboratory and clinical documents, ovarian ultrasound, and longitudinal
hormone/wearable/CGM streams — into standardized, traceable evidence, and then trains
*separate*, reusable representations for stable clinical phenotype, time-varying
hormonal state, ovarian morphology, confirmed symptom events, and parsed document
events. PCOS is the first condition-specific use case, but the data contracts and
encoders are deliberately not PCOS-shaped; condition-specific logic is confined to
`models/adapters/pcos/`.

## ⚠️ Non-diagnostic safety statement

**PRISM does not diagnose any condition.** It is a research artifact. It produces
research phenotype profiles, evidence summaries, morphology measurements,
model-estimated probabilities, missing-information analyses, and stability analyses.

It does **not** produce confirmed diagnoses, medical advice, treatment recommendations,
required clinical tests, guaranteed subtypes, or validated clinical decision support.
Every imaging measurement stays labelled *model-generated* until a clinician confirms
it, and every speech- or document-extracted item stays *unconfirmed* until a human
reviews it. Nothing here is validated for clinical deployment.

## The one scientific constraint that shapes everything

> **Do not pretend that unrelated datasets belong to the same patients.**

The static PCOS cohort, mcPHASES, NHANES, the ultrasound data, and the speech
evaluation set describe **different people**. They may train separate modules. They are
never randomly combined into artificial multimodal patients for training or validation.
This is enforced in code, not just in prose — see
[ADR-002](docs/decisions/ADR-002-no-fake-pairing.md).

## Architecture overview

```text
                          RAW INPUTS
  ┌────────────────────────────────────────────────────────┐
  │ Tabular clinical data                                  │
  │ Spoken symptoms or clinician dictation                 │
  │ Laboratory and clinical documents                      │
  │ Ovarian ultrasound images or volumes                   │
  │ Longitudinal hormones, wearables, CGM and symptoms     │
  └────────────────────────────────────────────────────────┘
                              │
                              ▼
                DATASET REGISTRY + INGESTION
                              │
                              ▼
               UNIVERSAL HORMONAL EVENT STORE
       values · time · units · modality · provenance
       confidence · confirmation · missingness · evidence
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
 STATIC CLINICAL PATH   UNSTRUCTURED PATH    TEMPORAL/IMAGE PATH
        │                     │                     │
        ▼                     ▼                     ▼
 Static baselines       Speech extraction     Ultrasound model
 Domain scores          Document parsing      Temporal GRU
 Tabular embedding      Confirmed events      Morphology/state
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    STANDARDIZED TOKENS
    static · symptom · document · ultrasound · temporal state
                              │
                              ▼
                   PCOS PHENOTYPE ADAPTER
          domain profiles · soft subtype similarity
          indeterminate output · confidence · stability
                              │
                              ▼
                    RESEARCH MODEL OUTPUT
       observed evidence · model-organized phenotype
       missing evidence · uncertainty · abstention
```

The architecture stops there. No clinician directory, phone script, PDF report, or
treatment layer is implemented until the evidence, modeling, stability, and validation
core is working and documented.

## Scope

Implemented in this repository (Steps 1–9):

| Step | Module | What it produces |
|:--|:--|:--|
| 1 | Repository, schemas, contracts | Typed contracts at every module boundary |
| 2 | Dataset registry and ingestion | Universal events + reproducible manifests |
| 3 | Static PCOS baseline | Calibrated, leakage-free held-out metrics |
| 4 | Continuous phenotype domains | Coverage-aware domain scores + learned embedding |
| 5 | Subtype and stability engine | Soft profiles, stability, abstention |
| 6 | Speech pipeline | Confirmed symptom events with evidence spans |
| 7 | Document pipeline | Grounded lab events, original + canonical values |
| 8 | Ultrasound pipeline | Ovary/follicle segmentation and morphology |
| 9 | Dynamic hormonal state | Current-state representation from longitudinal data |

Deliberately **excluded** for now: clinician contact search, insurance filtering, phone
scripts, doctor-facing PDFs, treatment recommendations, care navigation, a trained
cross-dataset fusion model, and any claim of clinical diagnosis or deployment.

## Installation

```bash
git clone https://github.com/AngelaNing1/Hack-Nation.git
cd Hack-Nation

make install
```

`make install` prefers [uv](https://github.com/astral-sh/uv) and falls back to a plain
`venv` + `pip install -e ".[dev]"`. Heavy or restricted dependencies (torch, xgboost,
pydicom, whisper, pdfplumber) are **optional extras** — every module degrades to a
documented fallback and the full test suite passes without them.

## Quick start

```bash
make install     # environment
make validate    # registry + data contracts
make test        # unit + contract tests
make smoke       # tiny synthetic end-to-end model runs
```

Everything above runs on committed synthetic fixtures. No clinical dataset is required,
and none is committed to this repository.

## Example pipeline invocation

```bash
# Static baseline, fully config-driven and reproducible
python scripts/train_static_baselines.py \
  --config configs/experiments/exp_static_baselines.yaml

# Continuous phenotype domains + learned tabular embedding
python scripts/train_tabular_autoencoder.py \
  --config configs/experiments/exp_phenotype_domains.yaml

# Discover profiles and stress-test their stability
python scripts/discover_phenotypes.py \
  --config configs/experiments/exp_subtype_stability.yaml
python scripts/run_stability_analysis.py \
  --config configs/experiments/exp_subtype_stability.yaml

# Unstructured evidence
python scripts/evaluate_speech_pipeline.py   --config configs/experiments/exp_speech_extraction.yaml
python scripts/evaluate_document_pipeline.py --config configs/experiments/exp_document_extraction.yaml

# Imaging and longitudinal state
python scripts/train_ultrasound.py --config configs/experiments/exp_ultrasound.yaml
python scripts/train_temporal.py   --config configs/experiments/exp_dynamic_state.yaml
```

## Dataset access

No clinical data lives in this repository. The registry at
[`registry/datasets.yaml`](registry/datasets.yaml) describes each dataset, how to obtain
it, what it may validly support, and what it must never be used to claim. Ingestion
adapters call `DatasetRegistry.require(dataset_id, use)` and **fail closed** on a use the
registry prohibits.

| Dataset | Access | Longitudinal | Primary use here |
|:--|:--|:--|:--|
| Public PCOS tabular cohort | External download | No | Static baseline, domains, clustering |
| mcPHASES | Restricted (PhysioNet) | Yes | Temporal hormonal-state model |
| NHANES 2021–2023 | Public (CDC) | No | Population reference, unit harmonization |
| USOVA3D | External download | No | Ovary/follicle segmentation and morphology |
| PRISM speech eval set | In-repo, synthetic | No | Speech extraction evaluation |
| PRISM document eval set | In-repo, synthetic | No | Document extraction evaluation |

## Repository map

```text
schemas/      typed contracts for every module boundary
registry/     datasets, canonical variables, units, phenotype domains, versions
ingestion/    one adapter per modality: load -> validate -> transform -> manifest
event_store/  append-only patient evidence store, conflicts preserved not resolved
features/     static features, missingness, transparent domain scores, manifests
models/       tabular · phenotype · stability · temporal · ultrasound · speech · documents
              adapters/pcos/  <- the ONLY place PCOS-specific logic lives
training/     splits, seeding, fold engine, checkpoints, experiment tracking
evaluation/   classification, calibration, clustering, stability, per-modality metrics
scripts/      config-driven entry points, one per experiment
tests/        unit · contract · integration · smoke, all on synthetic fixtures
docs/         concepts, datasets, modules, experiments, ADRs
artifacts/    splits, metrics, checkpoints, figures, model cards, manifests
```

## Implementation status

<!-- AUTO-GENERATED: IMPLEMENTATION-STATUS START -->
| Step | Component | Status | Independently validated? |
|:--|:--|:--|:--|
| 1 | Schemas and registries | Implemented | Contract tests |
| 2 | Ingestion + event store | Implemented | Unit tests, synthetic fixtures |
| 3 | Static baselines | Implemented | Cross-validated on the public cohort |
| 4 | Phenotype domains | Implemented | Reconstruction vs mean-imputation baseline |
| 5 | Subtype + stability | Implemented | Stability metrics only; no external validation |
| 6 | Speech pipeline | Implemented | Synthetic scripted corpus only |
| 7 | Document pipeline | Implemented | Synthetic report corpus only |
| 8 | Ultrasound pipeline | **Not implemented** | Segmentation/counting metrics on labelled data |
| 9 | Dynamic hormonal state | Implemented | Grouped participant-level held-out evaluation |
| — | Cross-modal fusion | **Not implemented** | Requires genuinely matched multimodal patients |
<!-- AUTO-GENERATED: IMPLEMENTATION-STATUS END -->

"Independently validated" means the module was evaluated on held-out data *from its own
dataset*. No claim of cross-module or clinical validation is made anywhere.

## Reproducibility

Every experiment is reproducible from a git commit, a configuration file, a random seed,
a dataset version, a split manifest, an environment lockfile, a model checkpoint, and a
metrics artifact. Each run writes a complete artifact directory:

```text
artifacts/experiments/<experiment_id>/
├── config.resolved.yaml    ├── metrics.json
├── environment.json        ├── predictions.parquet
├── git_commit.txt          ├── checkpoint/
├── data_manifest.json      ├── figures/
├── split_manifest.json     ├── model_card.json
├── feature_manifest.json   └── README.md
└── training_log.jsonl
```

All preprocessing is fitted **inside** the training fold. All longitudinal splits are
grouped by participant. Both properties have regression tests that fail loudly if broken.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use PRISM in research, please also cite the
underlying datasets under their own terms and reproduce their access restrictions.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
