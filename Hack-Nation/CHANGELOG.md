# Changelog

All notable changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

Schema changes are always listed explicitly — a schema is never changed
silently (see `registry/schema_versions.yaml`).

## [Unreleased]

### Fixed — `cycle_length` meant two different things, and the model learned the wrong one

The Kottarathil 2020 column `Cycle length(days)` was mapped onto canonical
`cycle_length`, which `registry/variables.yaml` defines as days from one menses
onset to the next (~28, range 1–365). The column's values centre near 5: the
fitted static scaler reports mean 4.94, scale 1.42. It is the duration of
bleeding. The static encoder therefore learned that feature under one meaning
while the registry, the mcPHASES temporal pipeline, and any patient-facing form
used the other — a real cycle length of 52 days scored 0.003 where 5 scored
0.985, a 300× inversion, and it would have surfaced only as "the model says low
risk for obviously PMOS-presenting patients".

The trained artifact is frozen and is not refit, so the correction is made at
the ingestion and encoder boundaries rather than in the checkpoint:

- **`registry/variables.yaml`** gains `menses_duration` (reproductive, days,
  range 1–14), documented as distinct from and easily confused with
  `cycle_length`.
- **`ingestion/tabular_pmos/mapping.py`** maps `Cycle length(days)` →
  `menses_duration`, with its unit and `menstrual_history` modality declared, so
  the value is validated against 1–14 instead of 1–365 and does not silently
  default to the `questionnaire` modality.
- **`models/tabular/encoder.py`** gains `LEGACY_FEATURE_ALIASES`. The artifact's
  slot keeps its trained name `cycle_length` and is fed from `menses_duration`
  on both the single-patient and the batch path. A frame carrying *both*
  variables drops the one named after the slot rather than letting it win, since
  that is precisely the value that must not reach it.
- **True `cycle_length` is now absent from the static branch** and is
  median-imputed; ovulatory dysfunction comes from the Rotterdam axis rules,
  which read cycle irregularity correctly.

`tests/unit/test_static_encoder_aliases.py` pins the bridge against the shipped
artifact: `menses_duration` moves the score, `cycle_length` is inert across
2–52, and the batch and single-patient paths agree.

### Changed — androgenic evidence is split, and profiles must earn their label

The androgenic axis was one domain mixing cutaneous signs with androgen assays.
That put the weights of assays that were never drawn into the coverage
denominator, so a patient with *recorded* hirsutism and acne still fell under
the 0.25 coverage floor and the whole androgenic axis abstained — for all 541
patients in the static cohort. Evidence that existed was being discarded by an
accounting artifact.

- **`registry/phenotype_domains.yaml` 1.1.0 → 1.2.0.** `androgenic` is replaced
  by `clinical_androgenic_evidence` (report-class only: `ferriman_gallwey_score`,
  `hirsutism`, `acne`, `androgenic_alopecia`, `hair_growth_face`) and
  `biochemical_androgenic_evidence` (assay-only: `total_testosterone`,
  `free_testosterone`, `dheas`, `shbg`). Each domain now declares an
  `evidence_source` of `symptoms`, `biochemical`, `imaging` or `mixed`.
  On separate denominators, clinical coverage is 1.5/3.5 = 0.43 and is
  assessable for all 541 patients; biochemical coverage is 0/4.5 and is
  unavailable for all 541.
- **`skin_darkening` moved from the androgenic domain to `metabolic`.**
  Acanthosis nigricans is a sign of insulin resistance, not androgen excess.
- **Combined androgenic evidence source, always stated:** `symptoms_only`,
  `biochemical_only`, `both`, or `unavailable`. Never omitted, never inferred.
  On this cohort it is always `symptoms_only`.
- **Defining-domain eligibility gate** in
  `models/adapters/pmos/prototype_similarity.py`. `metabolic_leaning` requires
  `metabolic`; `lh_amh_leaning` requires `lh_amh_pattern`;
  `androgenic_leaning` requires clinical *or* biochemical androgenic evidence;
  `mixed` requires at least two assessable domains. Ineligible profiles are
  removed before scoring and the remaining similarities renormalized — never
  zero-filled, never scored and then suppressed.
- **`dominant_profile` requires a stability verdict.** It is populated only when
  the stability engine calls the assignment stable; otherwise it is `null` with
  `indeterminate = true` and a reason. Never-checked is not stable.

### Does this alter a scientific claim?

Yes, in both directions and deliberately. Symptom-based androgenic evidence is
now reported where it exists instead of being silently dropped — but it is
labelled `symptoms_only` and is never presented as biochemical
hyperandrogenism. Conversely, no patient can be labelled `androgenic_leaning`
without androgenic input, and an unstable assignment no longer ships with a
profile name attached. Continuous domain scores are the primary phenotype
output; profile similarities are secondary and exploratory, not validated
clinical subtypes.

### Changed — calibration protocol tightened

`evaluation/calibration.py` keeps AUROC, AUPRC and Brier, and replaces the
reliability reporting with **five equal-frequency bins**. Each bin reports its
patient count, a 95% Wilson binomial interval on the observed rate, and an
`interpretable` flag that is `false` below 20 patients — because calibration
in a sparsely represented score range is not estimated, it is guessed.

A new `PlattCalibrator` **refuses to fit on anything but out-of-fold training
predictions** (`source="train_out_of_fold"`). In
`scripts/train_static_encoder.py` it is fitted on out-of-fold predictions from
the 432 training patients, then applied once, frozen, to the 109 held-out
patients. Nothing is fitted on held-out labels. Both `raw_model_score` and
`calibrated_model_score` are preserved.

Measured on the held-out split: AUROC 0.8927, AUPRC 0.8536, raw Brier 0.1270,
calibrated Brier 0.1147 (n=109; 432 training; Platt coefficient 0.814,
intercept −0.649).

### Changed — BREAKING: ultrasound is now 2D-primary

The ultrasound module was built 3D-first. That did not match clinical reality:
routine PMOS assessment uses **2D transvaginal imaging**, and the 2023
international guideline is written around follicle number per ovary, follicle
number per cross-section, and ovarian volume — it does not require a 3D
acquisition. USOVA3D was driving the design purely because it is one of the few
public datasets with expert ovary *and* per-follicle labels.

- **Input priority inverted.** 2D cine loop / multi-frame is now the primary
  pathway; a single 2D frame is a limited-output fallback; a 3D volume is an
  optional enhanced mode.
- **`schemas.imaging` 1.0.0 → 2.0.0 (not backward compatible).** The single
  `follicle_count` field is replaced by three non-interchangeable quantities —
  `follicle_number_per_section`, `estimated_follicle_number_per_ovary`, and
  `follicle_number_per_ovary` — plus `follicle_count_method`. Added
  `acquisition_mode`, `ovary_area_mm2`, `frames_analyzed`, `tracking_coverage`,
  and a `reportable_follicle_count` property that returns the count *with* its
  method.

  **Why it had to break:** one integer let a single cross-section silently
  claim a whole-ovary count. A `model_validator` now refuses any measurement the
  acquisition cannot support — a single frame cannot report a per-ovary count or
  an ovarian volume, and 2D frames cannot report a *true* per-ovary count.
- **Variable registry 1.0.0 → 1.1.0.** Added `follicle_number_per_section`,
  `estimated_follicle_number_per_ovary`, and `ovary_area_mm2`.
- **Dataset registry.** USOVA3D is reclassified as a pretraining/label resource
  and optional 3D benchmark, with a new prohibited claim of
  `independent_2d_evaluation` — a 2D test set carved from the same volumes used
  for pretraining is not independent. Added an `ovarian_ultrasound_2d`
  placeholder for the primary 2D pathway, which prohibits
  `follicle_instance_segmentation` because class-level labels cannot supervise
  per-follicle masks.
- **Modules.** `models/ultrasound/` reorganized around a 2D-primary layout
  (`qc_2d`, `ovary_detector_2d`, `segmenter_2d`, `cine_tracking`,
  `morphology_2d`) with the prior 3D work preserved as `segmenter_3d` /
  `morphology_3d`. Cine tracking matches follicles across adjacent frames so a
  follicle spanning frames 3–7 counts once, not five times.

### Does this alter a scientific claim?

Yes, and in the restrictive direction. The module previously could emit a
per-ovary follicle count from acquisitions incapable of supporting one. It now
reports the weaker quantity the data actually supports, labelled with the method
that produced it.

## [0.1.0] - 2026-07-18

Initial research preview covering Steps 1–9.

### Added

- **Schemas (1.0.0)** — `HormonalHealthEvent`, `ModalityToken`,
  `PatientSnapshot`, `EvidenceConflict`, `PhenotypeProfile`, `StabilityReport`,
  `OvarianMorphologyOutput`, `ParticipantDay`, `TemporalStateOutput`,
  `ExperimentResult`, `SplitManifest`.
- **Registries** — 6 datasets with allowed uses and prohibited claims, 65
  canonical variables, unit-conversion tables with per-factor tests, 4 phenotype
  domains, and a schema-version ledger.
- **Event store** — append-only storage, conflict detection that preserves both
  sides, provenance tracing, and parameterized model-ready snapshots.
- **Ingestion** — adapters for the public PMOS tabular cohort, NHANES, mcPHASES,
  speech, documents, and ultrasound.
- **Step 3** — static baselines (logistic regression, random forest, gradient
  boosting, MLP, majority-class and rule baselines) with repeated stratified
  patient-level cross-validation and full calibration reporting.
- **Step 4** — transparent coverage-aware phenotype-domain scores and a masked
  tabular autoencoder embedding.
- **Step 5** — clustering benchmark across representations, algorithms and
  K ∈ {2..6}; bootstrap, ablation and perturbation stability; indeterminate and
  abstention logic; hedged prototype mapping with a banned-phrase guard.
- **Step 6** — speech pipeline with offline scripted transcription, rule-based
  extraction with negation/temporality/uncertainty handling, evidence-span
  linking, a confirmation state machine, and a synthetic scripted corpus.
- **Step 7** — document pipeline with table extraction, registry-driven unit
  normalization preserving original and canonical values, reference-range
  parsing, page grounding, and a synthetic report corpus.
- **Step 8** — ultrasound pipeline with de-identification checks, quality gating
  with abstention, ovary/follicle segmentation, instance extraction, and
  morphology measurement in physical units.
- **Step 9** — GRU current-state model with hormone, cycle-phase, symptom and
  masked-reconstruction heads, grouped participant-level splits, and
  missing-modality ablations.
- **Repository** — CI (including a no-optional-dependencies job and a
  no-clinical-data guard), data-contract validation, model smoke tests, MkDocs
  documentation, ADRs 001–004, and issue/PR templates.

### Security

- `.gitignore`, a pre-commit hook, and a CI job independently block raw clinical
  data, imaging, audio and credentials from being committed.

### Known limitations

- No cross-modal fusion model. The datasets describe different people; see
  ADR-002.
- Speech and document metrics come from synthetic corpora and support no claim
  about real-world performance.
- No external validation of discovered phenotype profiles.
- Nothing in this release is validated for clinical use.
