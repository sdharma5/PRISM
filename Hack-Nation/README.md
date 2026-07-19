<h1 align="center">PRISM</h1>
<p align="center"><b>Platform for Reusable, Interpretable, Structured Multimodal Evidence</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research_preview-black?style=flat-square" alt="status: research preview">
  <img src="https://img.shields.io/badge/license-Apache_2.0-black?style=flat-square" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/python-3.11%2B-black?style=flat-square" alt="python 3.11+">
  <img src="https://img.shields.io/badge/next.js-14-black?style=flat-square" alt="Next.js 14">
  <img src="https://img.shields.io/badge/not_for-clinical_use-red?style=flat-square" alt="not for clinical use">
</p>

---

Built for the **6th Global AI Hackathon** — Hack-Nation in collaboration with MIT Club of Northern California and MIT Club of Germany.

Challenge track: *Building the AI Infrastructure for the Next Generation of Women's Health.*

---

## The problem

Women represent more than half of the global population, yet female physiology remains one of the least studied domains in AI and biomedical research. Conditions like PMOS, endometriosis, and menopause-related disease take years to diagnose on average. Hormonal health data — wearables, labs, ultrasound, patient voice, clinical documents — exists but is scattered, unstandardized, and rarely combined. There is no shared benchmark, no universal event schema, and no open multimodal infrastructure for researchers to build on.

PRISM contributes all three layers the hackathon calls for: a data and benchmark layer, a model layer, and a patient-facing application — all under Apache 2.0.

---

## What PRISM does

PRISM takes multiple forms of patient input — a clinical questionnaire, lab reports, wearable streams, spoken symptom narration, and ovarian ultrasound — and converts each one into a standardized, traceable evidence event. Seven independently trained model branches then process those events into a structured phenotype profile, a current hormonal state estimate, and a gap analysis of what evidence is missing and why it matters.

The first condition-specific adapter targets PMOS, implementing the 2023 International Evidence-based Guideline's Rotterdam criteria in transparent, auditable rules. The data contracts, event store, and model encoders are deliberately not PMOS-shaped; adding endometriosis or menopause tracking requires writing a new adapter in `models/adapters/`, not rewriting the pipeline.

---

## Non-diagnostic safety statement

**PRISM does not diagnose any condition.** It is a research artifact. Every imaging measurement remains labeled *model-generated* until a clinician confirms it. Every speech- or document-extracted item remains *unconfirmed* until a human reviews it. The system is not validated for clinical deployment and makes no diagnostic claims. Nothing in this repository constitutes medical advice.

---

## The one constraint that shapes everything

> Do not pretend that unrelated datasets describe the same patients.

The static PMOS cohort, mcPHASES longitudinal data, NHANES population reference, the USOVA3D ultrasound volumes, and the synthetic evaluation corpora describe different people. They train separate modules. They are never randomly combined into artificial multimodal patients for training or evaluation. This is enforced in code, not only in prose — `inference/orchestrator.py` hard-codes `joint_model_used=False`, and requesting `combination_mode="calibrated"` raises a `ValueError`. See [ADR-002](docs/decisions/ADR-002-no-fake-pairing.md).

---

## Architecture

```
                        RAW INPUTS
  ┌──────────────────────────────────────────────────────────┐
  │  Structured clinical questionnaire                       │
  │  Spoken symptom narration or clinician dictation         │
  │  Laboratory results and clinical documents (PDF)         │
  │  Ovarian ultrasound images or 3D volumes                 │
  │  Longitudinal hormones, wearables, CGM, symptom streams  │
  └──────────────────────────────────────────────────────────┘
                              │
                              ▼
              DATASET REGISTRY + INGESTION ADAPTERS
         validate · harmonize · unit-check · manifest
                              │
                              ▼
             UNIVERSAL HORMONAL EVIDENCE EVENT STORE
      value · time · unit · modality · provenance · confidence
      confirmation gate · missingness state · conflict flag
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
  STATIC CLINICAL        UNSTRUCTURED          TEMPORAL / IMAGE
        │                    │                    │
  Static baselines    Speech extraction     Ultrasound DualHeadUNet
  Domain scores       Document parsing      Temporal GRU state
  Tabular embedding   Confirmed events      Morphology estimates
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                   STANDARDIZED MODALITY TOKENS
      static · symptom · document · ultrasound · temporal-state
                             │
                             ▼
                  PMOS PHENOTYPE ADAPTER
         Rotterdam axes · domain profiles · soft subtype
         indeterminate output · stability · abstention
                             │
                             ▼
                  STRUCTURED RESEARCH OUTPUT
      observed evidence · missing evidence · uncertainty
      phenotype profile · gap analysis · ask-your-doctor list
                             │
                             ▼
                  PATIENT-FACING APPLICATION
      intake form · overview · evidence timeline · find care
      visit summary PDF · recommendations
```

No cross-modal fusion model exists. The seven branches never see each other's training data, and their outputs are combined by auditable rules, not by a jointly learned model. The application renders exactly what the rules produce, including abstentions and indeterminate assignments.

---

## Repository structure

```
Hack-Nation/
├── schemas/           Typed Pydantic contracts at every module boundary
├── registry/          Dataset permissions, 69 canonical variables, 7 phenotype domains
├── ingestion/         One adapter per modality — load, validate, transform, manifest
├── event_store/       Append-only patient evidence ledger; conflicts preserved, not resolved
├── features/          Static feature matrices, missingness indicators, domain scores
├── models/
│   ├── tabular/       Logistic regression baselines + masked autoencoder embedding
│   ├── phenotype/     Domain scores, clustering, prototype similarity
│   ├── stability/     Bootstrap, ablation, perturbation stability engine
│   ├── speech/        Rule-based symptom extraction from transcribed audio
│   ├── documents/     Rule-based lab event grounding from PDF reports
│   ├── ultrasound/    DualHeadUNet 3D segmentation and follicle counting
│   ├── temporal/      GRU current-state representation from longitudinal streams
│   └── adapters/pmos/ The only place PMOS-specific logic lives
├── training/          Splits, seeding, fold engine, checkpoints, experiment tracking
├── evaluation/        Classification, calibration, clustering, stability, per-modality
├── inference/         Orchestrator, evidence coordinator, presentation layer
├── scripts/           25+ config-driven entry points, one per experiment
├── apps/api/          FastAPI REST API (6 routers)
├── tests/             Unit, contract, integration, smoke — all on synthetic fixtures
├── docs/              Concepts, datasets, modules, experiments, ADRs
├── artifacts/         Splits, metrics, checkpoints, figures, model cards, manifests
├── datasets/          Dataset manifests and checksums (no clinical data committed)
└── UI/prism-app/      Next.js 14 patient-facing application
```

---

## Datasets

No clinical data is committed to this repository. The registry at [`registry/datasets.yaml`](registry/datasets.yaml) describes every dataset, how to obtain it, what claims it can validly support, and what it must never be used to assert. Ingestion adapters call `DatasetRegistry.require(dataset_id, use)` and fail closed on a prohibited use.

| Dataset | Source | License | Type | Primary use in PRISM |
|:--|:--|:--|:--|:--|
| Public PMOS tabular cohort (Kottarathil 2020) | Kaggle / UCI | CC0 public domain | Real, cross-sectional | Static baseline, domain scores, clustering |
| mcPHASES | PhysioNet (credentialed + DUA) | PhysioNet Restricted | Real, longitudinal | Temporal hormonal-state model |
| NHANES 2021–2023 | CDC | US Government (public domain) | Real, cross-sectional | Population reference, unit harmonization |
| USOVA3D | External download | See dataset page | Real, imaging | Ovary and follicle segmentation, morphology |
| PRISM speech evaluation corpus | In-repo | Apache 2.0 | Synthetic (scripted) | Speech extraction evaluation only |
| PRISM document evaluation corpus | In-repo | Apache 2.0 | Synthetic (scripted) | Document extraction evaluation only |

### Dataset notes

**PMOS tabular cohort (Kottarathil 2020):** 541 patients from 10 hospitals in Kerala, India; 177 PMOS-positive, 364 negative. CC0 — no access restrictions. One column labeled "Cycle length(days)" was confirmed to measure bleeding duration, not cycle length. PRISM corrects this in `ingestion/pmos_tabular/adapter.py` via `LEGACY_FEATURE_ALIASES` and documents it in `registry/variables.yaml`. This is not silently corrected — the adapter logs a warning on every run.

**mcPHASES:** 42 participants with daily urinary hormone measurements (LH, E3G, PDG), Fitbit wearable data, continuous glucose monitoring, menstrual cycle tracking, sleep, and symptoms. Credentialed PhysioNet access and a signed data-use agreement are required. The three longitudinal variables available for modeling are `urinary_lh`, `e3g`, and `pdg`.

**NHANES 2021–2023:** Population-representative survey with reproductive health, thyroid hormones, laboratory values, nutrition, and demographics. Survey design weights must be applied for any population-level inference; PRISM's ingestion adapter enforces this at the schema level.

**USOVA3D:** Sixteen annotated 3D ovarian ultrasound volumes (12 train, 2 validation, 2 test). The ultrasound branch is trained and evaluated but gated off in inference (`validated_for_inference: false` in the model registry) pending prospective clinical evaluation.

---

## Canonical variable registry

PRISM tracks 69 canonical variables organized into clinical domains. Every variable carries a canonical name, unit, valid range, evidence class, and missingness rules. The registry is the single source of truth for variable naming across all modules, the API, and the frontend. TypeScript types are auto-generated from it.

| Domain | Variables |
|:--|:--|
| Demographic | age |
| Anthropometric | bmi, height, weight, waist\_circumference, hip\_circumference |
| Reproductive / ovulatory | cycle\_length, menses\_duration, cycle\_irregularity, menstrual\_frequency\_per\_year, amenorrhea, lh\_fsh\_ratio, anti\_mullerian\_hormone, luteinizing\_hormone, follicle\_stimulating\_hormone, and 9 others |
| Clinical androgenic evidence | hirsutism, acne, androgenic\_alopecia, ferriman\_gallwey\_score, hair\_growth\_face |
| Biochemical androgenic evidence | total\_testosterone, free\_testosterone, dheas, shbg |
| Metabolic | fasting\_glucose, fasting\_insulin, hdl\_cholesterol, triglycerides, systolic\_blood\_pressure, diastolic\_blood\_pressure, waist\_hip\_ratio, and others |
| Ovarian morphology | follicle\_number\_per\_ovary, ovary\_volume\_ml, ovary\_area\_cm2, and others |
| Longitudinal | urinary\_lh, e3g, pdg, cycle\_phase, menstrual\_flow |
| Wearable | resting\_heart\_rate, hrv\_rmssd, sleep\_duration\_hours, activity\_steps, skin\_temperature |
| CGM | cgm\_mean\_glucose, cgm\_glucose\_sd, cgm\_time\_in\_range |
| Symptom | fatigue, weight\_gain, mood\_change, pelvic\_pain |
| History | family\_history\_pmos, family\_history\_diabetes |

Missing is never zero. PRISM enforces six distinct missingness states at the schema level: `observed`, `not_collected`, `collected_below_detection`, `not_applicable`, `derived`, and `imputed`. Imputed values are always labeled as such in outputs.

---

## Seven phenotype domains

The model computes a standardized domain score for each of the seven domains below. Scores are standard deviations from the training cohort mean — not percentages, not probabilities. A domain is marked *not assessed* when observed feature coverage falls below its minimum threshold.

| Domain | Min coverage | Features | Evidence class |
|:--|:--|:--|:--|
| Reproductive / ovulatory | 34% | 9 | Cycle and hormonal measurement |
| Metabolic | 34% | 12 | Anthropometric, glycemic, lipid |
| Clinical androgenic evidence | 25% | 5 | Cutaneous signs |
| Biochemical androgenic evidence | 25% | 4 | Androgen assays |
| Ovarian / LH-AMH | 25% | 7 | Hormonal and imaging |
| LH-AMH reproductive pattern | 34% | 5 | Hormonal pattern |
| Symptom burden | 34% | 8 | Patient-reported |

---

## Model architecture

### Static clinical branch

Logistic regression trained on the public PMOS cohort with stratified cross-validation. Calibrated by Platt scaling. Produces a PMOS-related model score (not a diagnosis probability), calibration metadata, feature coverage percentage, and imputation flag. This is the only branch permitted to issue a whole-patient PMOS score — the temporal and ultrasound branches are never aggregated into a single score.

Preprocessing is fitted inside the training fold. No information leaks across the train/test boundary. Both properties have regression tests that fail if violated.

### Tabular masked autoencoder

Trained on the same PMOS cohort with random feature masking. Produces a continuous embedding of clinical phenotype and per-domain coverage-aware scores. Reconstruction error against mean imputation baselines is reported in `artifacts/experiments/exp_phenotype_domains/metrics.json`.

### Subtype and stability engine

Soft phenotype profiles are derived by computing cosine similarity between a patient's domain score vector and a library of prototype vectors discovered by clustering. Profiles are exploratory similarities to patterns described in the literature — they are not validated clinical subtypes and are never reported as diagnoses.

The stability engine runs three independent checks before naming any profile:

- **Bootstrap resampling:** 200 resamples with replacement. Profile flip rate measured.
- **Feature-group ablation:** Each domain removed in turn. Assignment compared against full-feature result.
- **Score perturbation:** Gaussian noise added at the magnitude of measurement uncertainty. Stability score computed.

If the top two profile similarities are within the indifference threshold, or if fewer than the minimum required domains are observed, the output is *indeterminate*. Abstention is a first-class output, not a fallback.

### Speech pipeline

Rule-based symptom encoder. Zero learned parameters. Accepts transcribed audio (Whisper or any ASR output) and extracts symptom events with evidence spans, negation detection, uncertainty flags, and historical/current distinction. Evaluated on a synthetic 88-utterance scripted corpus. No real patient audio is used for training or evaluation.

### Document pipeline

Rule-based lab event encoder. Zero learned parameters. Parses PDF laboratory reports and clinical documents, extracts lab values with original units and canonical mappings, and grounds each value to a canonical variable code. Evaluated on 25 synthetic lab report documents.

### Ultrasound branch — DualHeadUNet

3D segmentation model with two output heads: ovary boundary segmentation and follicle detection. Trained on 12 USOVA3D volumes, validated on 2. Produces ovary volume, ovary area, follicle count per ovary, and PCOM morphology flag. This branch is trained and the metrics are documented in `artifacts/experiments/exp_ultrasound/`, but inference is gated off (`validated_for_inference: false`) until prospective clinical validation is complete.

### Temporal hormonal state branch

GRU-based encoder trained on mcPHASES longitudinal hormone streams (urinary LH, E3G, PDG). Produces a current-state representation and cycle-phase estimate from a sequence of daily measurements. Evaluated on held-out participants with grouped participant-level splits. This branch never yields a PMOS score — it contributes a state estimate that informs cycle-phase context.

---

## Rotterdam criteria implementation

PRISM implements the 2023 International Evidence-based Guideline for the Assessment and Management of PMOS. The Rotterdam criteria require two of three axes:

| Axis | Evidence sources | PRISM behavior |
|:--|:--|:--|
| Ovulatory dysfunction | Cycle length, irregularity, frequency per year, amenorrhea | Met / Not met |
| Hyperandrogenism — clinical | Hirsutism (Ferriman-Gallwey), acne, androgenic alopecia | Met / Not met — always carries qualifier when symptoms-only |
| Hyperandrogenism — biochemical | Total testosterone, free testosterone, DHEAS | Not assessable when no assay drawn — never inferred from symptoms |
| Polycystic ovarian morphology | Follicle number per ovary, ovary volume | Not assessable when no ultrasound — never inferred from labs |

Biochemical and morphology axes are marked *not assessable*, not *not met*, when the required measurements are absent. This distinction is enforced at the schema level and surfaced explicitly in the UI.

---

## API

FastAPI application. Start with:

```bash
python3 -m uvicorn apps.api.main:app --reload --port 8000
```

Startup loads all configured model encoders once and holds them for the process lifetime. If the static branch cannot load, startup fails — a service without it cannot issue a PMOS score, and starting anyway would mean every request returned a silently degraded result.

### Endpoints

| Method | Path | Description |
|:--|:--|:--|
| GET | `/api/v1/health` | Liveness check |
| GET | `/api/v1/models/status` | Per-branch availability and version |
| POST | `/api/v1/patients/infer` | Full inference across all available branches |
| POST | `/api/v1/patients/infer/static` | Static branch only |
| POST | `/api/v1/patients/infer/temporal` | Longitudinal branch only |
| POST | `/api/v1/patients/infer/ultrasound` | Returns 503 with reason (gated) |
| POST | `/api/v1/events` | Append events to the evidence ledger |
| GET | `/api/v1/events/{patient_id}` | Read patient events (unknown patient returns `[]`, not 404) |
| POST | `/api/v1/jobs/documents` | Submit a document ingestion job |
| POST | `/api/v1/jobs/speech` | Submit a speech ingestion job |
| GET | `/api/v1/jobs/{job_id}` | Poll job status |
| GET | `/api/v1/intake/schema` | Intake form field definitions with units and ranges |

Full request and response schemas are in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) and [`docs/openapi.yaml`](docs/openapi.yaml).

### Inference invariants enforced in code

- Temporal input alone never produces a PMOS score
- Ultrasound input alone never produces a PMOS score
- An indeterminate phenotype assignment never names a dominant profile
- A symptoms-only androgenic result always carries its qualifier
- `joint_model_used` is always `false`; requesting `combination_mode="calibrated"` raises `ValueError`

### Environment variables

| Variable | Default | Description |
|:--|:--|:--|
| `PRISM_DATA_ROOT` | — | Root directory for externally obtained datasets |
| `PRISM_ARTIFACT_ROOT` | — | Output directory for experiment artifacts |
| `PRISM_CORS_ORIGINS` | `http://localhost:3000,http://localhost:3001` | Comma-separated allowed origins |
| `PRISM_EVENT_LOG` | — | Path to mirror the event ledger to JSONL (unset = in-memory) |
| `PRISM_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Frontend application

Next.js 14 / React 18 / TypeScript patient-facing application. Source at `UI/prism-app/`.

### Pages

| Route | Description |
|:--|:--|
| `/intake` | Structured clinical questionnaire — all fields sourced from the variable registry with units and valid ranges |
| `/overview` | PMOS evidence profile: model score, Rotterdam axes, domain scores, phenotype profile, stability, missing evidence |
| `/timeline` | Chronological evidence ledger with per-event confirmation, provenance, extraction confidence, and conflict flags |
| `/care` | Specialist finder, insurance plan context, derived question list for the next clinical appointment, printable visit summary |
| `/recommendations` | Domain-weighted lifestyle and clinical priority list |
| `/research` | Full disclosure: what the models are, what data they were trained on, what claims they do and do not support |

### Running the frontend

```bash
cd UI/prism-app
npm install
npm run dev
# Opens at http://localhost:3000
```

TypeScript types are auto-generated from the API's OpenAPI spec:

```bash
npm run gen:types
```

### Key design decisions

**Intake form:** Every field label, unit, and valid range is served from the API at `/api/v1/intake/schema`. The frontend never hardcodes variable names. A hardcoded unit is an undetectable hundredfold error; this approach makes unit errors a test failure instead.

**Confirmation gates:** Every event extracted by the speech or document pipeline is held in `awaiting_patient_confirmation` or `awaiting_clinician_confirmation` state. It does not contribute to any model score until a human reviews and accepts it. The timeline page surfaces this explicitly.

**Evidence transparency:** Every displayed value shows its modality, provenance, extraction confidence, and missingness status. Nothing is presented as a clean number without its source.

**Abstention rendering:** Indeterminate phenotype assignments, not-assessable Rotterdam axes, and unassessed domains are displayed as first-class states, not hidden or collapsed. The UI is designed around the assumption that not knowing is a valid and common output.

**Visit summary:** A printable PDF-style summary is generated client-side, including the patient's intake responses, Rotterdam axis results, domain scores, and a derived list of tests and questions to discuss at the next appointment.

---

## Installation

```bash
git clone https://github.com/your-org/prism.git
cd prism/Hack-Nation

# Preferred (uses uv if available, falls back to pip)
make install

# Or manually
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Heavy dependencies (PyTorch for ultrasound, Whisper for speech, pdfplumber for documents, XGBoost) are optional extras. Every module degrades gracefully to a documented fallback when they are absent. The full test suite passes without any of them.

```bash
# Install with specific extras
pip install -e ".[api]"          # FastAPI + uvicorn
pip install -e ".[imaging]"      # PyTorch + pydicom + scikit-image
pip install -e ".[speech]"       # Whisper
pip install -e ".[documents]"    # pdfplumber
pip install -e ".[dev]"          # All of the above + test dependencies
```

---

## Quick start

```bash
make install     # Set up environment
make validate    # Validate registry and data contracts
make test        # Run unit and contract tests
make smoke       # Tiny synthetic end-to-end runs of all model branches
```

All of the above run on committed synthetic fixtures. No clinical dataset is required. No dataset is committed to this repository.

---

## Training pipelines

All scripts use a shared CLI defined in `scripts/_cli.py`. Flag resolution order: explicit CLI flag > environment variable > config file > built-in default.

```bash
# Static baseline
python scripts/train_static_baselines.py \
  --config configs/experiments/exp_static_baselines.yaml

# Phenotype domains and tabular embedding
python scripts/train_tabular_autoencoder.py \
  --config configs/experiments/exp_phenotype_domains.yaml

# Subtype discovery and stability validation
python scripts/discover_phenotypes.py \
  --config configs/experiments/exp_subtype_stability.yaml
python scripts/run_stability_analysis.py \
  --config configs/experiments/exp_subtype_stability.yaml

# Speech and document extraction evaluation
python scripts/evaluate_speech_pipeline.py \
  --config configs/experiments/exp_speech_extraction.yaml
python scripts/evaluate_document_pipeline.py \
  --config configs/experiments/exp_document_extraction.yaml

# Ultrasound segmentation
python scripts/train_ultrasound.py \
  --config configs/experiments/exp_ultrasound.yaml

# Longitudinal temporal state
python scripts/train_temporal.py \
  --config configs/experiments/exp_dynamic_state.yaml
```

SLURM job templates for each experiment are in `slurm/`.

---

## Reproducibility

Every experiment artifact directory contains:

```
artifacts/experiments/<experiment_id>/
├── config.resolved.yaml      Complete resolved configuration
├── environment.json          Python version, package versions
├── git_commit.txt            Exact commit hash
├── data_manifest.json        Dataset version and checksums
├── split_manifest.json       Train/validation/test row indices
├── feature_manifest.json     Feature names, imputation flags
├── training_log.jsonl        Per-epoch metrics
├── metrics.json              Final evaluation metrics
├── predictions.parquet       Per-sample predictions and labels
├── checkpoint/               Serialized model weights and scaler
├── figures/                  Calibration curves, confusion matrices
├── model_card.json           Intended use, limitations, metrics
└── README.md                 Human-readable experiment summary
```

All preprocessing fitted inside the training fold. All longitudinal splits grouped by participant. Both properties have regression tests that fail loudly if broken.

---

## Implementation status

| Step | Component | Status | Validation |
|:--|:--|:--|:--|
| 1 | Schemas and registries | Complete | Contract tests |
| 2 | Ingestion and event store | Complete | Unit tests on synthetic fixtures |
| 3 | Static PMOS baseline | Complete | Cross-validated on the public cohort |
| 4 | Phenotype domain scores | Complete | Reconstruction vs mean-imputation baseline |
| 5 | Subtype and stability engine | Complete | Stability metrics; no external validation |
| 6 | Speech pipeline | Complete | Synthetic 88-utterance scripted corpus |
| 7 | Document pipeline | Complete | Synthetic 25-document corpus |
| 8 | Ultrasound pipeline | Complete, inference gated | Segmentation and counting metrics on labeled volumes |
| 9 | Temporal hormonal state | Complete | Grouped participant-level held-out evaluation |
| — | Cross-modal fusion model | Not implemented | Requires genuinely matched multimodal patients |

"Validated" means the module was evaluated on held-out data from its own dataset. No cross-module or clinical validation is claimed.

---

## Deployment

### Backend — Railway

1. Create a new project at railway.app and connect this repository.
2. Set the root directory to `Hack-Nation/`.
3. Set the start command to:
   ```
   python3 -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
   ```
4. Add environment variables: `PRISM_CORS_ORIGINS` set to your Vercel frontend URL.

### Frontend — Vercel

1. Import `UI/prism-app/` as a new Vercel project.
2. Set the root directory to `UI/prism-app`.
3. Set the environment variable `NEXT_PUBLIC_PRISM_API_URL` to your Railway backend URL.
4. Deploy. Vercel handles Next.js builds automatically.

---

## CI

GitHub Actions runs on every push:

| Workflow | What it checks |
|:--|:--|
| `ci.yml` | Unit, contract, and integration tests on Python 3.11, 3.12, 3.13 |
| `data-contracts.yml` | Registry consistency, variable mapping validity |
| `model-smoke-tests.yml` | End-to-end synthetic runs of all model branches |
| `docs.yml` | MkDocs build |
| `release.yml` | Release artifact packaging |

The CLI contract test (`tests/integration/test_cli_contract.py`) parses every command in this file, in `TRAINING.md`, in `docs/`, and in `slurm/*.sbatch`, and fails if any script name or flag does not exist. Documentation drift is a test failure.

---

## Tech stack

**Backend**

- Python 3.11+
- FastAPI 0.115+ with uvicorn
- Pydantic 2.7+ (schema validation)
- scikit-learn 1.4+ (baselines, preprocessing, clustering)
- numpy 1.26+, pandas 2.2+, scipy 1.12+
- PyTorch 2.2+ (ultrasound segmentation — optional)
- XGBoost 2.0+ (optional gradient boosting baselines)
- pydicom 2.4+, scikit-image 0.22+ (imaging — optional)
- pdfplumber 0.11+ (document parsing — optional)
- Whisper (speech transcription — optional)
- pytest, mypy (strict), ruff

**Frontend**

- Next.js 14.2.5 / React 18 / TypeScript 5
- Tailwind CSS 3.4.1
- Framer Motion 11.3.8
- Recharts 2.12.7
- Zustand 4.5.4
- Lucide React

---

## Open science

All source code, model checkpoints, experiment configurations, evaluation pipelines, and synthetic evaluation corpora are published under Apache 2.0. The variable registry, dataset manifests, and canonical schema definitions are included in the repository.

Datasets with their own access requirements (mcPHASES requires PhysioNet credentialing; USOVA3D has its own terms) are not redistributed but their processing pipelines, ingestion adapters, and derived manifests are fully open. A researcher with access to those datasets can reproduce every experiment from a single config file and a git commit hash.

Model cards for every trained branch are written to `artifacts/experiments/<id>/model_card.json` at training time and committed to the repository. They include intended use, training data, known limitations, evaluation metrics, and the specific claims the model does and does not support.

---

## Contributing

Contributions are welcome. Before opening a pull request:

```bash
make validate   # Registry and contract checks
make test       # Full test suite
```

The test suite runs on synthetic fixtures only. If you add a new modality, dataset, or model branch, add a synthetic fixture in `tests/fixtures/` and a smoke test in `tests/smoke/`. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

New condition-specific adapters go in `models/adapters/<condition>/`. The pipeline, event store, and encoders are condition-agnostic by design.

---

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Five-layer design and cross-cutting invariants
- [`DATASET_REGISTRY.md`](DATASET_REGISTRY.md) — Dataset access, permitted uses, canonical variables
- [`MODEL_CARD.md`](MODEL_CARD.md) — Limitations, ethical considerations, reproducibility notes
- [`TRAINING.md`](TRAINING.md) — How to train every model branch, locally and on SLURM
- [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) — Full API specification
- [`docs/decisions/`](docs/decisions/) — Architecture decision records
- [`docs/concepts/`](docs/concepts/) — Event store, confirmation gates, pairing constraints
- [`docs/datasets/`](docs/datasets/) — Per-dataset documentation

MkDocs site:

```bash
pip install mkdocs-material
mkdocs serve
```

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).

Datasets have their own licenses. The public PMOS cohort is CC0. NHANES is US Government public domain. mcPHASES requires a PhysioNet data-use agreement. USOVA3D terms are on the dataset page. Nothing in this repository redistributes restricted data.

---

## Citation

See [`CITATION.cff`](CITATION.cff). If you use PRISM in research, also cite the underlying datasets under their own terms and reproduce their access restrictions in any downstream work.

---

*6th Global AI Hackathon — Hack-Nation × MIT Club of Northern California × MIT Club of Germany*
*Women's Hormonal Health track — Building the AI Infrastructure for the Next Generation of Women's Health*
