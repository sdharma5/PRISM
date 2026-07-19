# PRISM — Complete Architecture Specification

**Purpose of this document.** A self-contained technical description of the PRISM system, written to be read by another language model with no access to the repository. Every claim below is sourced to a file path. Where the code and the documentation disagree, this document states the code's behavior and flags the discrepancy.

**Repository root:** `Hack-Nation/`
**Version:** 0.1.0, dated 2026-07-18, status "Research preview" (`MODEL_CARD.md:6`)

---

## 0. The one-paragraph summary

PRISM estimates evidence for polycystic ovary syndrome (PMOS) from up to five unrelated data channels — tabular clinical variables, menstrual-cycle time series, ovarian ultrasound, speech transcripts, and lab documents. **It is not one model.** It is seven independently trained modules whose outputs are wrapped in a shared envelope and combined by a deterministic weighted average with hand-chosen weights. There is no learned fusion model, and by design there cannot be one: the five datasets describe five disjoint populations, so no patient has more than one modality. In production only one branch is a learned classifier that emits a whole-patient probability (the tabular branch, a logistic regression), one branch is a learned forecaster (temporal), and one is trained but switched off (ultrasound). Speech and document "models" are deterministic rule-based encoders with zero learned parameters.

---

## 1. System topology

```
                    ┌─────────────────────────────────────────┐
                    │  registry/  (the authority)             │
                    │  datasets.yaml   — what may be claimed  │
                    │  variables.yaml  — ~65 canonical vars   │
                    │  units.yaml      — conversion factors   │
                    │  phenotype_domains.yaml                 │
                    └─────────────────────────────────────────┘
                                      │ governs
                                      ▼
  raw data → ingestion adapters → EVENT STORE (append-only) → snapshots → features
                                                                            │
        ┌───────────────────────────────────────────────────────────────────┤
        ▼                ▼               ▼             ▼              ▼
   static/tabular   temporal state   ultrasound    speech       documents
   (LogisticReg)    (LOCF+ridge)     (DualHeadUNet) (rule-based)  (rule-based)
        │                │               │             │              │
        └────────────────┴───────────────┴─────────────┴──────────────┘
                                      │
                          all emit one ModalityToken shape
                                      │
                                      ▼
                    EvidenceCoordinator  (weighted average,
                     weights fit to NOTHING — design rules)
                                      │
                                      ▼
                    5 domain scores + abstention + agreement
                                      │
                                      ▼
                    PMOS adapter (Rotterdam axis rules, the ONLY
                     PMOS-specific code in the repository)
                                      │
                                      ▼
                    WebsitePMOSProfileResponse
```

**Five architectural layers** (`ARCHITECTURE.md:18-21`): `ingestion → event store → features → models → tokens`.

**Core design thesis** (`ARCHITECTURE.md:3-14`): evidence arrives from incompatible channels at different times, units, and reliabilities. Most pipelines flatten to a wide table at ingestion. PRISM flattens *as late as possible* and makes flattening a parameterized, reproducible, inspectable operation.

---

## 2. The data substrate

### 2.1 Event store (ADR-001, `docs/decisions/ADR-001-event-store.md`)

The single evidence substrate is an append-only list of `HormonalHealthEvent`. Events are **never mutated or deleted**; a confirmation appends a revision. `conflict_resolution.py` **detects and labels conflicts but never picks a winner**. The wide table everyone else treats as the source of truth is here a derived, parameterized *view*, produced by:

```
build_snapshot(as_of, allowed_confirmation_statuses, modality_filter)
  → values + missingness mask + conflicts + exclusion reasons
```

Cost of this choice: every consumer must go through the snapshot API.

### 2.2 Human confirmation gate (ADR-004)

Each event carries a `provenance` and a `confirmation_status`. Provenances `document_extracted`, `model_measured`, and `model_inferred` **cannot** become `confirmed` without a `reviewed_by`. A model only ever sees events where `is_model_ready` is true (`confirmed` or `not_required`). Extraction quality is measured pre-confirmation; modeling uses post-confirmation data only.

### 2.3 The no-fake-pairing rule (ADR-002) — load-bearing

The five datasets describe **different people**. `patient_id` is scoped by `source_dataset`. Each modality trains an independent encoder. **No joint fusion model exists in the system**, and synthetic multimodal combinations may demo the UI but may never appear in any evaluation, metric table, or model card.

Enforcement is mechanical: `joint_model_used=False` is hard-coded (`inference/orchestrator.py:181`) and a schema validator makes `True` unreachable (`inference/report_schema.py:137-147`). Requesting `combination_mode="calibrated"` **raises `ValueError`** citing ADR-002 (`inference/evidence_coordinator.py:99-104`).

### 2.4 Datasets

| id | Real/synthetic | License | n | Notes |
|---|---|---|---|---|
| `pmos_tabular_public` | **Real** patient records, 10 hospitals, Kerala India (Kottarathil 2020) | CC0 public domain | 541 (177 pos / 364 neg) | The only labeled PMOS cohort |
| `nhanes_2021_2023` | Real survey | US Gov public domain | — | Reference ranges only; must use survey weights |
| `mcphases` | Real, **credentialed** PhysioNet + DUA | Restricted | 42 participants | Must live outside repo tree; currently checked in — a compliance violation |
| `usova3d` | Real ultrasound volumes | — | 16 volumes | 12 train / 2 val / 2 test |
| `prism_document_eval_synthetic`, `prism_speech_eval_synthetic` | **Synthetic** | — | 25 docs / 88 utterances | Evaluation fixtures only |

Restrictions are **executable**, not documentary: adapters call `DatasetRegistry.require(dataset_id, use)`, which raises `PermissionError` on a prohibited use. E.g. `pmos_tabular_public` prohibits `validated_four_subtype_classification` and `prospective_clinical_deployment`.

---

## 3. The token envelope

Every modality emits the same object. This is what makes the modules comparable.

```python
ModalityToken:
    patient_id, modality, embedding, structured_features,
    quality_score, confidence_score, observed_at, model_version,
    source_dataset, provenance_ids, missing_fields, warnings
```

**Critical semantic** (`ARCHITECTURE.md:92-106`): tokens share an envelope so they are **comparable, not concatenable**. Nothing ever concatenates embeddings across modalities.

---

## 4. MODULE 1 — Static / tabular branch (the only deployed PMOS classifier)

**File:** `models/tabular/encoder.py:98` — `StaticClinicalEncoder`
**Artifact:** `artifacts/encoders/static_clinical/static_clinical_encoder.joblib` (9,548 bytes)

This is the only branch entitled to emit a whole-patient P(PMOS). Three independent code paths enforce that (`configs`: `pmos_statement_requires_static: true`; `models/adapters/pmos/abstention.py:108-117`; `inference/presentation/website_mapper.py:188`).

### 4.1 Architecture — a three-step sklearn Pipeline

```python
Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("model",  LogisticRegression(max_iter=2000,
                                  class_weight="balanced",
                                  C=1.0, solver="lbfgs",
                                  random_state=0)),
])
```

- **Median, not mean**, because AMH and LH/FSH are right-skewed.
- **`class_weight="balanced"`** because the cohort is 364 negative / 177 positive.
- Fitted on **training rows only**; `training/engine.py:75` returns the pipeline unfitted so the caller cannot accidentally leak.

### 4.2 Exact input — 19 features, alphabetical order

Column order is frozen alphabetically so it never depends on CSV column order. Full fitted statistics from `artifacts/encoders/static_clinical/static_clinical_encoder.json`:

| # | Feature | Median (imputer) | Scaler mean | Scaler scale | LR coef |
|--:|---|--:|--:|--:|--:|
| 1 | age | 31.0000 | 31.5718 | 5.5000 | −0.1400 |
| 2 | anti_mullerian_hormone | 3.8250 | 5.4942 | 5.2578 | 0.4427 |
| 3 | bmi | 24.3209 | 24.4390 | 4.0056 | −0.0607 |
| 4 | **cycle_length** ⚠ | 5.0000 | **4.9398** | **1.4244** | −0.2988 |
| 5 | diastolic_blood_pressure | 80.0000 | 77.1296 | 4.7240 | 0.0200 |
| 6 | fasting_glucose | 98.0000 | 99.9519 | 19.8522 | 0.0077 |
| 7 | follicle_count_left | 5.0000 | 6.1181 | 4.2549 | **0.9098** |
| 8 | follicle_count_right | 6.0000 | 6.6250 | 4.4470 | **1.3346** |
| 9 | follicle_stimulating_hormone | 4.7950 | 5.2737 | 4.6060 | −0.1600 |
| 10 | height | 156.0000 | 156.4164 | 6.0395 | −0.2372 |
| 11 | hip_circumference | 96.5200 | 96.5024 | 9.9453 | 0.4704 |
| 12 | lh_fsh_ratio | 2.2049 | 3.6362 | 4.7409 | −0.1591 |
| 13 | luteinizing_hormone | 2.1900 | 2.7107 | 2.2828 | −0.0478 |
| 14 | progesterone | 0.3150 | 0.6049 | 4.0841 | −0.2744 |
| 15 | resting_heart_rate | 72.0000 | 73.5255 | 2.7452 | 0.1918 |
| 16 | systolic_blood_pressure | 110.0000 | 114.8148 | 5.8896 | −0.0025 |
| 17 | waist_circumference | 86.3600 | 86.0307 | 9.1097 | −0.2742 |
| 18 | waist_hip_ratio | 0.8947 | 0.8926 | 0.0463 | 0.1481 |
| 19 | weight | 60.0000 | 59.8977 | 10.9679 | 0.5382 |

**Intercept −0.5338.** The two follicle-count features dominate the model (coefs 0.91 and 1.33, roughly 2–3× the next largest). This is worth internalizing: **the deployed PMOS classifier is largely an antral-follicle-count model.**

### 4.3 ⚠ THE NAMED BUG — feature slot 4 is mislabeled

Documented at `MODEL_CARD.md:90-100`. This is the single most important defect in the system.

The Kottarathil source column `Cycle length(days)` was ingested as canonical `cycle_length`. But the fitted scaler shows **mean 4.94, scale 1.42** — those are *bleeding days*, i.e. **menses duration**, not the menses-to-menses interval the registry defines as `cycle_length`.

The artifact cannot be renamed without refitting, so the slot keeps its trained name and is fed from `menses_duration` through an alias table:

```python
# models/tabular/encoder.py:70
LEGACY_FEATURE_ALIASES = {"kottarathil-2020": {"cycle_length": "menses_duration"}}
```

**Consequences a reader must not get wrong:**
1. A true cycle length is **not an input** to the static branch. It stays absent and is median-imputed. Ovulatory dysfunction is instead assessed by Rotterdam axis rules downstream.
2. Feeding a real cycle length into that slot puts the patient **~33 standard deviations** out of distribution and **inverts the score**: 5 → 0.985, 52 → 0.003.
3. The batch path `predict_proba` (`encoder.py:227-248`) **drops before renaming**, so a frame carrying both the true variable and the slot-named variable discards the latter rather than letting it win the rename.

### 4.4 Output token

`export_token` (`encoder.py:252`) emits:
- `embedding = []` — **empty**. This encoder emits structured features, not a vector.
- `structured_features`: `{domain}_score`, `{domain}_coverage`, raw canonical variables, `pmos_evidence_probability`
- `quality_score` = observed feature fraction
- `confidence_score = abs(p − 0.5) * 2` — **distance from the decision boundary, not a probability**

### 4.5 Platt calibration (this head only)

`artifacts/encoders/static_clinical/calibrator.json`:

```
method: Platt    coef: 0.8136    intercept: −0.6491
n_fit: 432       fit_source: train_out_of_fold
```

Fitted on out-of-fold training predictions, then **frozen** and applied once. `PlattCalibrator.from_dict` rejects any payload not fitted out-of-fold. If `calibrator.json` is absent or malformed the system degrades softly — scores are reported raw and labelled `calibrated=False` (`apps/api/registry.py:286-311`).

Being a monotone transform, calibration does **not** change AUROC or AUPRC — only Brier and reliability.

### 4.6 Measured performance (deployed encoder)

`artifacts/encoders/static_clinical/metrics.json`, n_train 432, held-out n 109, positive rate 0.3303:

| Metric | Value |
|---|---|
| Held-out AUROC | **0.8927** |
| Held-out AUPRC | 0.8536 |
| Brier raw → calibrated | 0.1270 → **0.1147** |
| CV AUROC | 0.8898 ± 0.0268, folds [0.8396, 0.9019, 0.8855, 0.9107, 0.9113] |

All 5 reliability bins interpretable (n = 22/22/22/22/21). Calibration fixes bin 3 (gap −0.1804 → −0.0055) but leaves bin 4 at +0.0904.

### 4.7 Research-side comparators (not deployed)

`artifacts/experiments/exp_static_baselines_20260718T223209Z/`, 10 folds (5 × 2 seeds), n≈108/fold, prevalence 0.3272:

| Model | AUROC | AUPRC | Bal. acc | Brier | ECE | Calib. slope |
|---|---|---|---|---|---|---|
| majority_class | 0.5000 ± 0.0000 | 0.3272 | 0.5000 | 0.2201 | 0.0046 | 0.342 |
| single_feature_rule | 0.8830 ± 0.0259 | 0.7985 | 0.8197 | 0.1440 | 0.1600 | 2.175 |
| **static_logistic** | **0.9429 ± 0.0185** | 0.9198 | 0.8764 | 0.0867 | 0.0804 | **0.944** |
| static_random_forest | **0.9523 ± 0.0111** | 0.9270 | 0.8800 | 0.0875 | 0.1133 | 2.009 |
| static_xgboost | 0.9481 ± 0.0101 | 0.9195 | 0.8737 | **0.0799** | 0.0697 | 0.897 |
| static_mlp | 0.9155 ± 0.0507 | 0.8748 | 0.8242 | 0.1070 | 0.1105 | 1.611 |

**Sanity check the project imposes on itself** (`TRAINING.md:88`): the best learned model beats a single-feature threshold rule by only **+0.069 AUROC**. Random forest's calibration slope of 2.01 fails the project's own calibration check, which is why logistic (slope 0.944) is deployed despite marginally lower AUROC.

**Note the gap:** CV AUROC 0.9429 vs deployed held-out 0.8927. Different protocols; do not quote them interchangeably.

### 4.8 Other tabular models defined in code but not deployed

- `models/tabular/masked_autoencoder.py:26` — numpy denoising autoencoder, hand-written backprop + Adam. Layer widths `[2d → 64 → 16 → 64 → d]`, tanh, linear output. Input width is `2d` because values and observed-mask are concatenated: `hstack([X_in, mask_in])`. Loss = masked MSE over **artificially hidden observed cells only**, mask rate ∈ (0.1, 0.3). Measured: reconstruction MSE **0.7560 ± 0.2185** vs mean-imputation **1.0185** — beats the baseline on 100% of folds.
- `models/tabular/mlp.py:20` — sklearn MLP, hidden (64, 32), relu, alpha 1e-3, early stopping.
- `models/tabular/logistic.py:102` — RandomForest, 300 trees, `class_weight="balanced_subsample"`.
- `models/tabular/xgboost.py:30` — falls back to `HistGradientBoostingClassifier` when xgboost is absent, emits `RuntimeWarning`, records `backend` in the model card.
- `models/tabular/baselines.py` — majority-class and single-feature rule (grid over feature × sign × threshold, fit on training fold only).

---

## 5. MODULE 2 — Temporal branch

**Deployed:** `models/temporal/state_encoder.py:98` — `TemporalStateEncoder`
**Artifact:** `artifacts/encoders/temporal_state_v1/temporal_state_encoder.joblib` (27,015 bytes)

### 5.1 ⚠ The deployed temporal model is NOT a neural network

This is the most common misreading of the codebase. The deployed encoder selects a **different simple model per target**, chosen by frozen benchmark:

```python
DEFAULT_TARGET_MODELS = {"lh": "locf", "e3g": "ridge_window", "pdg": "locf"}
# cycle_phase → logistic_phase
```

- **LOCF** = last-observation-carried-forward. `translate_method` maps it to the user-facing string *"Based on the latest observed value"* — the docstring is explicit that LOCF must never be described as a prediction.
- **ridge_window** = ridge regression (alpha 10.0) on the flattened 14-day causal window.
- **logistic_phase** = `LogisticRegression(max_iter=1000, class_weight="balanced")`.

The GRU exists in code and was trained as an experiment, but is **rejected by the registry** (`inference/encoder_registry.py:112-124`) on the grounds that "the echo-state model's recurrent weights are fixed random projections and it has never been run on the frozen split; the GRU does not exist" as a persisted artifact.

### 5.2 Input representation

7 channels, fixed order: `lh, e3g, pdg, resting_heart_rate, wrist_temperature, hrv_rmssd, mean_glucose`.
Window layout: `[values(14 × 7), observed_mask(14 × 7)]` = **196 input dimensions**. `lookback_days = 14`, `MIN_DAYS_FOR_STATE = 14` — under 14 days the encoder emits an abstention token with `temporal_state_available=False`.

Normalization statistics (`normalization_stats.json`):

| Channel | mean | std | n_observed |
|---|--:|--:|--:|
| lh | 6.0711 | 7.7648 | 3154 |
| e3g | 134.5572 | 103.9573 | 3154 |
| pdg | 6.6954 | 7.6650 | **1058** |
| resting_heart_rate | 69.4479 | 5.6946 | 2861 |
| wrist_temperature | 33.8153 | 0.9226 | 2669 |
| hrv_rmssd | 54.6415 | 27.1164 | 2814 |
| mean_glucose | 111.8746 | 13.7006 | **1887** |

### 5.3 Uncertainty

Empirical residual quantiles from training predictions — **not a learned head**. `interval_coverage=0.8` → 10th/90th percentile bounds.

### 5.4 Measured performance

`artifacts/encoders/temporal_state_v1/benchmark_metrics.json`, split `mcphases_participant_split_v1`:

| Target | MAE | model | n |
|---|--:|---|--:|
| LH | **3.2653** | locf | 1083 |
| E3G | **50.6377** | ridge_window | 1083 |
| PdG | **3.2549** | locf | 393 |
| cycle phase | macro-F1 **0.5653**, bal. acc 0.5754 | logistic | 1134 |

Against baselines: global-mean gives LH 3.9601 / E3G 76.3481 / PdG 4.4993. **The ridge gain over plain LOCF for E3G is 52.09 → 50.64, i.e. 1.45 MAE (2.8%)** — the project's own `target_models.json` admits this is narrow.

### 5.5 The experimental GRU (trained, not deployed)

`models/temporal/state_model.py:158` — `TemporalStateModel`. Worth describing because it is what most readers assume is running.

**Feature construction** (`models/temporal/gru.py`), 3 columns per channel plus cycle encoding plus modality group block:
```python
n_features = 3 * len(channels) + 2 + len(MODALITY_GROUPS)
# per channel: (normalized_value, is_observed, log1p(staleness))
# + sin/cos of 2π·(day % 28)/28
# MODALITY_GROUPS = ("hormone", "wearable", "cgm", "symptom")
```
GRU-D style decay on carried values: `carried * exp(-0.35 * staleness)`.

**Encoder:** `NumpyGRU` (`gru.py:264`) with **fixed random weights, never trained** — an echo-state reservoir. Standard GRU equations, `(N, T, F) → (N, H)`, H=32. A `TorchGRU` twin and a dilated causal `TCN` (3 levels, dilations [1,2,4], receptive field 15 days) also exist.

**Four linear heads on the frozen embedding:** hormone reconstruction (closed-form ridge per channel), cycle state (4-way multinomial logistic, classes `menstrual/follicular/peri_ovulatory/luteal` — `"unknown"` is never predicted), symptom (per-symptom binary logistic, targeting **next-day** symptoms to avoid a copying task), masked reconstruction.

**Composite loss:** `L = 1.0·L_hormone + 1.0·L_cycle + 0.5·L_symptom + 0.5·L_masked`. Every term divides by an observed-mask sum, so an unobserved value is never scored against zero.

**Measured** (`artifacts/experiments/exp_dynamic_state/metrics.json`, 3 folds, 13 test participants): cycle-phase macro-F1 **0.4317 ± 0.0213**, accuracy 0.4382. Hormones are *better* than the deployed encoder (LH MAE 2.5052, E3G 30.0358, PdG 1.6173) — but on a different split, so not directly comparable. Symptoms are near-useless: macro-AUPRC 0.2514, and per-symptom F1 of **0.0** for breast_tenderness, 0.0246 mood_low, 0.0449 cramps.

**⚠ Ablation defect:** the `no_symptoms` ablation produces **bitwise-identical** metrics and coverage to `full` (macro-F1 0.4551 both). The symptom channel has no effect on the cycle-phase path — this reads as an unexercised ablation rather than a finding. Other ablations behave: `no_wearable` costs 0.2001 macro-F1, `sparse_hormones` 0.2413.

---

## 6. MODULE 3 — Ultrasound branch (TRAINED BUT GATED OFF)

**Model:** `models/ultrasound/dual_head_unet.py:61` — `DualHeadUNet(nn.Module)`
**Status:** `configs/models/inference_encoders.yaml:26` — **`enabled: false`**

### 6.1 Why it is off

The config states the checkpoint exists and loads cleanly, but the ovary segmentation head is uncorrected and the branch has no true end-to-end held-out evaluation — **its reported follicle Dice is oracle-assisted and is not a deployable patient result.** The encoder is not constructed at all, so the score is unreachable from the API. `POST /infer/ultrasound` **unconditionally raises 503** before doing any work (`apps/api/routers/inference.py:105-127`); a 200 with an empty body would invite a client to render "no findings," which is a clinical claim this branch cannot make.

### 6.2 Architecture

**Two independent sigmoid heads, not a 3-class softmax** — because a follicle is nested *inside* an ovary, so the classes are not mutually exclusive.

Conv block (`_conv_block:42`):
```python
nn.Sequential(
    conv(cin, cout, 3, padding=1), norm(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
    conv(cout, cout, 3, padding=1), norm(cout, affine=True), nn.LeakyReLU(0.01, inplace=True),
)
```

**`InstanceNorm`, never `BatchNorm`** — batches are 1–2 volumes. **No dropout anywhere in the repository.**

Config `dims=3, in_channels=1, base_channels=16, depth=3` (~1.5–1.9M params for 12 training volumes):

| Stage | Channels |
|---|---|
| enc[0] | 1 → 16, MaxPool(2) |
| enc[1] | 16 → 32, MaxPool(2) |
| enc[2] | 32 → 64, MaxPool(2) |
| bottleneck | 64 → 128 |
| decoder ×2 (weights **not** shared) | ConvT(128→64) + block(128→64); ConvT(64→32) + block(64→32); ConvT(32→16) + block(32→16) |
| heads ×2 | `Conv(16, 1, kernel=1)` |

Decoders are separate because ovary boundary and follicle interiors are different shapes at different scales.

**Input:** `(N, 1, D, H, W)` — dim1 is a single B-mode greyscale channel.
**Output:** two raw **logit** maps, `{"ovary": (N,1,D,H,W), "follicle": (N,1,D,H,W)}`. Sigmoid is applied at inference.

### 6.3 Loss

```
L = 1.0·L_ovary + 1.0·L_follicle + 0.1·L_outside
```
- **Ovary:** Dice + BCE-with-logits.
- **Follicle:** Dice + focal BCE, `γ=2.0`, **`α=0.75`** — above 0.5 because follicle voxels are a minority within a minority; 26% of label slices contain no follicle at all.
- **Outside penalty:** penalizes follicle probability where there is no ovary. Gradients **deliberately flow to both heads** — detaching would discard the informative "grow the ovary" gradient.
- `dice_loss` uses **`smooth = 1.0`, not an epsilon**, with an explicit rationale: at 1e-6 a correct near-zero prediction on an empty mask still scores ~1.0 loss.

At inference the anatomical constraint is applied again, not just in the loss: `follicle_mask = follicle_mask * ovary_mask` (`trained_encoder.py:287`).

### 6.4 Training hyperparameters

3D (`configs/experiments/exp_usova3d_3d_unet.yaml`): patch [32,96,96], foreground_fraction 0.5, **epochs 120, batch 2, lr 1e-3, weight_decay 1e-5, seed 42, early_stopping_patience 25**, monitor `val_follicle_dice` (max), mixed precision auto.
2D v2: patch [128,128], epochs 60, batch 8, patience 20.
Split: seed 42, grouped by `volume_id`, 12 train / 2 val / 2 test.

### 6.5 ⚠ Measured performance — the validation/test gap

| Run | best **val** follicle Dice | best epoch | **held-out test** follicle Dice |
|---|--:|--:|--:|
| 2D v1 | 0.9276 | **4** | 0.4910 |
| 2D v2 | 0.7462 | 60 | **no test_metrics.json — never evaluated** |
| 3D | 0.5966 | 65 | **0.5603** |

The v2 config header documents the cause: v1's protocol reported 0.93 validation Dice for a checkpoint whose held-out Dice was 0.49, and selected an **epoch-4** model. v2 averages 12 evenly-spaced samples per volume to fix the estimator — but was then never scored on the test volumes.

Held-out test, 2 volumes only (`test_metrics.json`):

| | 3D r1 | 2D v1 r1 |
|---|--:|--:|
| ovary_dice | 0.6051 | 0.6344 |
| follicle_dice | 0.5603 | 0.4910 |
| instance_precision | **0.1203** | **0.0417** |
| instance_recall | 0.3214 | 0.0714 |
| follicle_count_MAE | **12.0** | **30.5** |

**The follicle head massively over-segments.** Every 3D test volume shows ~470k–540k false follicle voxels outside the ovary. Vol110: 22 predicted follicles vs 8 annotated. Instance precision 0.045–0.12.

Stored caveats: two test volumes give point estimates with no meaningful CI (smoke-level); annotators agree on exact follicle count in only **5 of 16** volumes, so the r1/r2 gap is a lower bound on label uncertainty.

### 6.6 Supporting ultrasound code (all deterministic, zero learned weights)

`ovary_detector_2d.py`, `quality.py`, `qc_2d.py`, `morphology_2d/3d.py`, `follicle_instances.py`, `cine_tracking.py`. Quality gating is deterministic because **no image-quality labels exist in USOVA3D**. Thresholds: `MIN_FOLLICLE_DIAMETER_MM=2.0` (2023 guideline AFC definition) and `MIN_FOLLICLE_VOXELS=27` — the latter a scale-free floor added because without it an uncalibrated volume produced **649 "follicles" against 8 annotated**.

A torch-free `ThresholdSegmenter` fallback exists (Gaussian smooth → Otsu → closing → largest component) but requires **both** `implementation: "heuristic"` and `allow_heuristic_fallback: true`; both are false in the shipped config. A missing checkpoint is treated as a **configuration error, not a degraded mode**.

---

## 7. MODULES 4 & 5 — Speech and documents (ZERO learned parameters)

Both are deterministic encoders. This is deliberate: there is no paired corpus to train on, and an untrained neural encoder would emit numbers that *look* learned but mean nothing (`models/speech/event_embedding.py:3`).

### 7.1 Speech — `models/speech/symptom_encoder.py:34`

Multi-hot over a lexicon. Dimension = `len(vocabulary) * 3 + 16`.

**Three channels — present / negated / historical** at offsets `0`, `width`, `2*width`, because *"denies acne"* and *"had acne years ago"* must not cancel or alias each other. Unknown codes fall into 16 sha256-addressed hash buckets.

```python
confidence = mean(extraction_confidences)
quality = clip(audio_quality, 0, 1) * confidence   # joint ceiling
```
Bad audio caps a confident extraction and vice versa.

**Measured** (`artifacts/metrics/speech_extraction/`, 88 utterances, synthetic): most fields F1 1.0; the one real failure is `coverage_limit` F1 **0.2857** (precision 1.0, recall 0.1667). WER is 0.0 **by construction**. The notes state that scores near 1.0 are "NOT evidence of a good extractor" — this is a regression guard, not a performance estimate.

### 7.2 Documents — `models/documents/evidence_encoder.py:48`

Fixed 17-analyte vector. Embedding = **value channel ⊕ presence mask**, length **34**:
```python
embedding = [numeric_channel(code) for code in FEATURE_ORDER]      # 17
presence_mask = [1.0 if code in structured else 0.0 ...]           # 17
```
A missing analyte reads 0.0 in the value channel; **the mask is the only thing distinguishing it from a genuine zero.**

**Measured** (25 synthetic documents): F1 **1.0 on every field**, page_grounding_accuracy 1.0 (66/66), unsupported_value_rate 0.0. Synthetic — supports no real-world claim.

### 7.3 ⚠ Neither is wired into the API

`ModelRegistry.load` (`apps/api/registry.py:183-189`) constructs the orchestrator **without `event_extractors`**. Any speech or document input therefore triggers a warning at `inference/orchestrator.py:82-87` and its content **never reaches a model**.

---

## 8. MODULE 6 — Phenotype clustering and stability

### 8.1 Clustering (`models/phenotype/clustering.py`)

Algorithms: kmeans, gaussian_mixture, agglomerative, spectral (off by default — O(n³), unstable at this n), consensus. K ∈ {2..6}, seeds 0..4. Consensus builds a co-association matrix over 40 resamples at 80% subsample, then average-linkage on `1 − consensus`.

**Selected configuration** (`artifacts/experiments/exp_subtype_stability/discovery_summary.json`): 177 label-positive participants, **K=2**, representation `domain_scores`, algorithm `kmeans`, from 60 scored configurations. Silhouette **0.3054**, bootstrap Jaccard **0.9040**, cross-seed ARI **0.9689**.

Note K=2, not 4. ADR-003 states **"Four is never a default"** — a direct guard against reproducing the four-subtype PMOS literature by assumption.

### 8.2 Hedged language is mechanically enforced (ADR-003)

Post-hoc enrichment labels may use only *resembles / most similar to / has overlap with*. A banned-phrase list is enforced in `models/phenotype/prototype_mapping.py`. Stability travels with every assignment.

`PrototypeSimilarityModel` (`models/adapters/pmos/prototype_similarity.py:190`): cosine similarity to **literature-declared centroids** (not fitted), softmax at temperature 0.25. `affinities` is documented as **normalized affinity scores, NOT calibrated probabilities**. Ineligible profiles are removed and renormalized over, never zero-filled.

### 8.3 Temperature-scaled membership calibration

`models/stability/calibration.py` — distinct from the static branch's Platt scaling.

- `temperature_scale` = `log(p)/T` renormalized — **monotone, so argmax is preserved.** A calibration step that silently reassigned patients would be indefensible.
- `fit_temperature` minimizes ECE between top-1 membership and the participant's **bootstrap agreement rate** — not classification correctness. Reverts to T=1.0 if ECE does not improve.

### 8.4 ⚠ Measured stability is weak

`artifacts/experiments/exp_phenotype_validation/phenotype_validation.json`, n=541, 100 bootstraps at noise 0.25:

- ARI **0.4129 ± 0.0286**, NMI 0.4016
- per-patient agreement 0.7468, with **28.8% of patients below 0.6 agreement**
- `indeterminate_fraction` **0.5333** — over half of patients get no stable dominant profile
- `biochemical_androgenic_evidence` observation rate **0.0 — never observed for any patient**

A dominant profile is reported only when **both determinate AND stable** (`website_mapper.py:328-333`).

---

## 9. FUSION — deterministic, not learned

**File:** `inference/evidence_coordinator.py:49`

### 9.1 The formula

For each domain *d*:

```
S_d = Σ_m (w_md · q_m · c_m · s_md) / Σ_m (w_md · q_m · c_m)
```

where `w` = design-rule relevance weight from `configs/models/evidence_coordination.yaml`, `q` = token quality, `c` = token confidence, `s` = mapped domain score.

**`w` was fit to nothing.** It is a table of human judgments. A weight of `0.0` means **excluded from that domain entirely**, not down-weighted — e.g. `ovarian_ultrasound → metabolic: 0.0`, `static_clinical → current_state: 0.0`.

**Five domains:** `reproductive, androgenic, metabolic, ovarian_morphology, current_state`.

### 9.2 Abstention is a first-class output

If `mass < min_domain_evidence_mass` (0.20), the domain returns `level="insufficient_evidence"` with **`score=None`**. A schema validator *forbids* a score alongside an abstaining level. Throughout the codebase, an unmeasured quantity is `None`, never `0.0` — because 0.0 in a z-scored domain means "exactly average," which is a measurement.

### 9.3 Agreement, not voting

`classify_agreement` bands by spread: `strong ≤0.15`, `moderate ≤0.30`, else `conflicting`. **Conflicts are surfaced as notes; the score is still computed and never averaged away silently.** `PmosFeatureMapper` records cross-modal conflicts as `EvidenceConflict` with `requires_human_review=True` and keeps the **first** value so the outcome does not depend on dict iteration order.

### 9.4 Evidence bands

`_EVIDENCE_BANDS`: `<0.25 low`, `<0.50 moderate`, `<0.75 elevated`, else `high`. `None → "not_available"` — explicitly **never** `low`, because *"low evidence is a finding, not available is the absence of one."* Bands rather than a raw percentage because "the calibration is only trustworthy where the reliability bins are populated."

---

## 10. The PMOS adapter — the only condition-specific code

`models/adapters/pmos/` is explicitly **"the only PMOS-specific code in the repository"** (`ARCHITECTURE.md:70-90`). The event store, encoders, and token envelope know about *variables*, not about PMOS. This is the condition-agnostic guarantee: another condition would be a new adapter, not a new pipeline.

The adapter applies **Rotterdam criteria axis rules** (`diagnostic_features.py:331-427`), each axis reporting `status ∈ {met, not_met, uncertain, not_assessable}` with `threshold_sources` citing the guideline. Domain scores are weight-weighted means of observed z-scores; when coverage falls below `min_coverage_to_report` the score is **withheld, not caveated** — *"a number on a page outlives its footnote."*

⚠ Because the tabular cohort contains **no androgen assay**, the biochemical androgenic domain is unavailable for **every** patient. Androgenic evidence is always symptoms-only.

---

## 11. Inference request path

`POST /api/v1/patients/infer`

| # | Step | Location |
|--:|---|---|
| 1 | Route handler | `apps/api/routers/inference.py:48-62` |
| 2 | Pydantic validation, `extra="forbid"`; rejects an all-null feature map as an unfilled template | `apps/api/schemas/requests.py:37-95` |
| 3 | Each non-null feature → `HormonalHealthEvent` (`modality="questionnaire"`, `provenance="patient_confirmed"`). **Nulls skipped, never encoded as `value=None`** | `requests.py:158-190` |
| 4 | `_one_patient_only` validator rejects foreign `patient_id`s | `inference/patient_bundle.py:99-130` |
| 5 | Orchestrator branch gating | `inference/orchestrator.py:101-186` |
| 6 | Static encoder → token | `models/tabular/encoder.py:252` |
| 7 | Temporal encoder → token | `models/temporal/state_encoder.py:336` |
| 8 | `EvidenceCoordinator.combine` | `inference/evidence_coordinator.py:76-172` |
| 9 | PMOS adapter → `PMOSProfileOutput` | `models/adapters/pmos/evidence_adapter.py:227-411` |
| 10 | Platt calibration applied (frozen) | `evidence_adapter.py:203-207` |
| 11 | `to_website_response` | `inference/presentation/website_mapper.py:125-162` |

**Branch gating** is the orchestrator's *only* decision. Encoders are **injected, not imported**, so the module has no dependency on torch or on any encoder being trained. Every branch is wrapped in a bare `except Exception` that converts failure into a warning string — one failing encoder must not deny the patient the report the others support.

**Feature encoding boundary.** Unknown codes have three distinct fates:
1. Not in `feature_names` → ignored for the model row.
2. In `registry/variables.yaml` → carried into `structured_features` so guideline axes can threshold variables the model never learned (`hirsutism`, `acne`, `total_testosterone`, `ferriman_gallwey_score`).
3. Not in the registry → **dropped rather than travelling on as evidence.**

`_NON_FEATURE = {"patient_id", "pmos_binary"}` is excluded so the **training label can never ride inside a token**.

**No unit conversion happens on this path** — `convert_to_canonical` is an ingestion-layer function. Values submitted via `clinical_features` are assumed canonical.

---

## 12. Response object

`WebsitePMOSProfileResponse` (`apps/api/schemas/responses.py:298-349`), `extra="forbid"`, `RESPONSE_SCHEMA_VERSION = "1.0.0"`.

Key fields: `report_id` (= `rpt_<sha256(patient_id|timestamp)[:16]>`, hashed so it does not leak the patient id into logs), `modality_coverage`, `pmos_assessment`, `rotterdam_axes`, `phenotype`, `current_state`, `supporting_evidence`, `conflicting_evidence`, `missing_evidence`, `provenance`, `warnings`.

**The honesty ledger**, carried in the payload: `learned_components_used` (only ever `static_clinical.pmos_head`) vs `rule_based_components_used` (`evidence_coordinator.design_rule_weights`, `ultrasound.pcom_threshold_rules`, `pmos_adapter.guideline_axes`).

`is_diagnosis: Literal[False]` — not overridable. `disclaimer` is carried **in the payload** so any client, including one nobody here wrote, receives it. `clinician_review_status = "model_generated"`.

---

## 13. Cross-cutting invariants

Each has a failing regression test (`ARCHITECTURE.md:108-125`):

1. **No preprocessing leakage** — pipelines returned unfitted; scalers fit on training rows only.
2. **Patient-level splits** — never day-level or row-level.
3. **Missing is never zero** — six distinct missingness states.
4. **Unconfirmed evidence never reaches a model** — `is_model_ready`.
5. **Model-generated ≠ measured** — `clinician_review_status`.
6. **Hedged language** — banned-phrase guard.
7. **Abstention is a valid output.**

Engineering patterns worth noting:
- **Dropout appears nowhere.** Regularization is L2/weight decay, early stopping, small capacity, and input masking.
- **Missingness is always a channel**: `(value, is_observed, staleness)` triples in temporal, `hstack([values, mask])` in the autoencoder, `embedding + presence_mask` in documents.
- **Every loss divides by an observed-mask sum**, so an unobserved value is never scored against zero.
- All heavy dependencies (torch, xgboost, pydicom, scikit-image, whisper, pdfplumber) are **optional with real fallbacks**; CI runs the full suite without extras.

---

## 14. What is gated off in production

| # | Thing | Mechanism |
|--:|---|---|
| 1 | Ultrasound branch | `enabled: false`; endpoint unconditionally 503s |
| 2 | `combination_mode="calibrated"` | **raises `ValueError`** citing ADR-002 |
| 3 | `joint_model_used=True` | schema-unreachable |
| 4 | Heuristic segmenter fallback | needs two opt-in flags, both false |
| 5 | Temporal `echo_state` / `trained_gru` | rejected — not persisted or benchmarked |
| 6 | Speech/document extractors | not wired into `ModelRegistry.load` |
| 7 | Startup without static branch | **fails hard** (`require_static=True`) |
| 8 | CORS | explicit allowlist, not `*`; GET/POST/OPTIONS only |

**Deliberate stopping point** (`ARCHITECTURE.md:135-139`): no clinician directory, insurance filter, phone script, doctor PDF, treatment recommendation, or care navigation — *"downstream of a validated core, and the core is not validated yet."*

---

## 15. ⚠ Known defects — consolidated

### Documented in the model card
1. **The `cycle_length` slot holds menses duration** (§4.3). Inverts the score if misused. Highest-severity item in the system.
2. Static cohort is single-site, cross-sectional, modest n, label not re-adjudicated.
3. No androgen assay in the cohort → biochemical androgenic domain unavailable for all 541 patients.
4. Phenotype profiles are exploratory similarities, not validated subtypes; prototype centroids are **declared from literature, not fitted**; biochemical androgen thresholds are **placeholders**.
5. Speech/document metrics come from **synthetic** corpora and support no real-world claim.
6. **No cross-modal fusion, no prospective evaluation, no clinical validation of any kind.**

### Found in artifacts, NOT written into any card
7. **`artifacts/model_cards/static_logistic.md` is stale and describes a different run** — it cites dataset version `synthetic-fixture-v1`, n=60/fold, prevalence 0.4633, **AUROC 0.6360 ± 0.0506**. The real cohort run gives **0.9429**. The published card for the deployed model reports synthetic-fixture numbers.
8. **2D v2 checkpoints carry `model_version: 'ultrasound-usova3d-3d-v1'` while `model_config.dims == 2`** — a mislabeled 2D checkpoint under a 3D version string.
9. **`exp_usova3d_2d_unet_v2` has no `test_metrics.json`** — the corrected-protocol run was never scored on held-out volumes.
10. **The phenotype card's "Dice 0.49 / instance F1 0.05" is v1's number**; the deployed 3D model scores 0.5603 / 0.1750. The card is stale.
11. **`no_symptoms` ablation is a no-op** — bitwise-identical to `full`.
12. **`TRAINING.md:85` promises "≥5 seeds"** but the shipped configs use 2 seeds (static) and 1 seed (phenotype).
13. **No held-out set exists for either PMOS experiment** — `holdout_ids` is empty in both manifests. 432+109 = 541 and 405+136 = 541, so every patient rotates through a test fold. `training/splits.py` supports the parameter; it was not used. The model card's phrase "held-out patients" means the rotating CV test fold, not an untouched partition.
14. **mcPHASES is checked into the repo** (`datasets/mcphases/`, including a 367 MB zip) despite the registry requiring it to live outside the tree under `PRISM_DATA_ROOT` — a DUA compliance issue.

### Previously fixed, preserved as regression context
- Temporal days identify their subject as `participant_id`, not `patient_id`; a foreign day once passed the identity check and was scored into another person's report.
- Pooled missing counts made coverage exceed 1.0, reporting "0% observed" for a patient who supplied 5 of 19 features.
- Trusting a `_score` suffix admitted `ferriman_gallwey_score` (raw value 9.0) into a dict where everything else is a cohort z-score, skewing similarity and stability.
- The quality gate's unsafe-acceptance rate was 67% during development; it must stay at 0.0.

---

## 16. Summary table — what is actually learned

| Module | Learned? | Deployed? | Serialized artifact | Headline metric |
|---|---|---|---|---|
| Static clinical | **Yes** — LogisticRegression | **Yes** | 9.5 KB joblib | Held-out AUROC **0.8927** |
| Static baselines (RF/XGB/MLP) | Yes | No | none (metrics only) | CV AUROC 0.92–0.95 |
| Masked autoencoder | Yes | No | none | MSE 0.756 vs 1.019 baseline |
| Temporal state (deployed) | **Partly** — ridge + logistic; LOCF is not learned | **Yes** | 27 KB joblib | LH MAE 3.27, phase F1 0.565 |
| Temporal GRU | Echo-state: **encoder is fixed random**, heads learned | No | none | phase macro-F1 0.432 |
| Ultrasound DualHeadUNet | **Yes** | **No — gated** | 7.7 MB .pt (3D) | test follicle Dice 0.560, instance F1 0.175 |
| Phenotype clustering | Yes (unsupervised) | Yes | CSV/JSON | K=2, silhouette 0.305, ARI 0.413 |
| Speech encoder | **No — zero parameters** | Not wired | none | F1 1.0 synthetic |
| Document encoder | **No — zero parameters** | Not wired | none | F1 1.0 synthetic |
| **Fusion** | **No — hand-set weights** | Yes | none | **no metric exists** |

**The single most important sentence for a reader to carry away:** the only learned component that produces a whole-patient PMOS probability is a 19-feature logistic regression dominated by two antral-follicle-count variables, and everything above it in the stack — the domain scores, the cross-modal weights, the Rotterdam axes — is a deterministic rule system whose parameters were chosen by humans and fit to nothing.
