# Dataset registry

No clinical data is committed to this repository. This document describes each
dataset, how to obtain it, what it may validly support, and what it must never
be used to claim.

The machine-readable source of truth is
[`registry/datasets.yaml`](registry/datasets.yaml). Ingestion adapters call
`DatasetRegistry.require(dataset_id, use)` and **raise `PermissionError`** on a
use the registry prohibits — the restrictions below are executable, not advisory.

## Registered datasets

<!-- AUTO-GENERATED: DATASET-REGISTRY START -->
| Dataset | Access | Longitudinal | Allowed uses | Prohibited claims |
|:--|:--|:--|:--|:--|
| **mcphases**<br><sub>mcPHASES</sub> | restricted_physionet | Yes | `temporal_state_model`<br>`cycle_phase_prediction`<br>`hormone_reconstruction`<br>`symptom_forecasting`<br>`missingness_analysis` | `pmos_subtype_validation`<br>`pmos_diagnosis`<br>`binary_baseline` |
| **nhanes_2021_2023**<br><sub>NHANES 2021-2023</sub> | public_cdc_download | No | `population_reference`<br>`unit_harmonization`<br>`external_stress_testing`<br>`missingness_analysis` | `longitudinal_state_modeling`<br>`validated_pmos_subtypes`<br>`pmos_diagnosis` |
| **ovarian_ultrasound_2d**<br><sub>2D ovarian ultrasound dataset (to be selected)</sub> | external_download_pending_selection | No | `ovary_classification`<br>`image_quality_modeling`<br>`ultrasound_embedding`<br>`pcom_morphology_classification`<br>`external_stress_testing` | `pmos_diagnosis`<br>`validated_pmos_subtypes`<br>`follicle_instance_segmentation`<br>`prospective_clinical_deployment` |
| **pmos_tabular_public**<br><sub>Public PMOS clinical tabular dataset</sub> | public_or_external_download | No | `binary_baseline`<br>`phenotype_domain_modeling`<br>`exploratory_clustering`<br>`missingness_analysis` | `validated_four_subtype_classification`<br>`prospective_clinical_deployment`<br>`longitudinal_state_modeling` |
| **prism_document_eval_synthetic**<br><sub>PRISM synthetic laboratory-report evaluation set</sub> | in_repository_synthetic | No | `document_extraction_evaluation`<br>`unit_conversion_evaluation`<br>`page_grounding_evaluation` | `clinical_document_validation`<br>`real_world_performance_estimate` |
| **prism_speech_eval_synthetic**<br><sub>PRISM scripted speech evaluation set (synthetic)</sub> | in_repository_synthetic | No | `speech_extraction_evaluation`<br>`negation_evaluation`<br>`temporality_evaluation` | `clinical_speech_validation`<br>`real_world_performance_estimate` |
| **usova3d**<br><sub>USOVA3D</sub> | public_or_external_download | No | `ovary_segmentation`<br>`follicle_segmentation`<br>`follicle_counting`<br>`morphology_measurement`<br>`slice_extraction_for_2d_pretraining`<br>`optional_3d_benchmark` | `pmos_diagnosis`<br>`validated_pmos_subtypes`<br>`prospective_clinical_deployment`<br>`independent_2d_evaluation` |
<!-- AUTO-GENERATED: DATASET-REGISTRY END -->

## The pairing constraint

These datasets describe **different people**. They may train separate modules.
They are never combined into artificial multimodal patients for training or
validation. See [ADR-002](docs/decisions/ADR-002-no-fake-pairing.md).

## Access notes

**Public PMOS tabular cohort.** External download. Cross-sectional, single-site,
modest sample size. Its ultrasound columns are numbers transcribed from reports,
not measurements taken from images — they are treated as
`derived_ultrasound_measurements`, distinct from the imaging module's output.

**mcPHASES.** Restricted; requires credentialed PhysioNet access and a data-use
agreement. Store it outside the repository tree under `PRISM_DATA_ROOT`. This is
the only genuinely longitudinal dataset here and therefore the only one allowed
to support the temporal state model.

**NHANES 2021–2023.** Public CDC download with a complex survey design. Any
population estimate must use the survey weights carried through by
`ingestion/nhanes/survey_design.py`. Unweighted use is limited to reference
ranges and unit harmonization.

**USOVA3D.** External download. Segmentation ground truth is annotator-derived,
so imaging metrics inherit annotator variability. Morphology measurements depend
on physical spacing being present; where it is absent, the module abstains
rather than reporting a volume in voxels.

**PRISM synthetic corpora.** The speech and document evaluation sets are written
for this repository, committed, and entirely fictional. They exercise the
pipelines and support **no** claim about real-world performance.

## Canonical variables

<!-- AUTO-GENERATED: VARIABLE-REGISTRY START -->
Total canonical variables: **69**

| Domain | Count | Variables |
|:--|--:|:--|
| anthropometric | 5 | `bmi`, `height`, `hip_circumference`, `waist_circumference`, `weight` |
| biochemical_androgenic_evidence | 4 | `dheas`, `free_testosterone`, `shbg`, `total_testosterone` |
| cgm | 3 | `cgm_glucose_sd`, `cgm_mean_glucose`, `cgm_time_in_range` |
| clinical_androgenic_evidence | 5 | `acne`, `androgenic_alopecia`, `ferriman_gallwey_score`, `hair_growth_face`, `hirsutism` |
| demographic | 1 | `age` |
| history | 2 | `family_history_diabetes`, `family_history_pmos` |
| label | 1 | `pmos_binary` |
| longitudinal | 5 | `cycle_phase`, `e3g`, `menstrual_flow`, `pdg`, `urinary_lh` |
| medication | 1 | `medication_current` |
| metabolic | 17 | `bmi`, `cgm_glucose_sd`, `cgm_mean_glucose`, `cgm_time_in_range`, `diastolic_blood_pressure`, `fasting_glucose`, `fasting_insulin`, `hdl_cholesterol`, `hip_circumference`, `homa_ir`, `ldl_cholesterol`, `skin_darkening`, `systolic_blood_pressure`, `triglycerides`, `waist_circumference`, `waist_hip_ratio`, `weight_gain` |
| ovarian | 4 | `anti_mullerian_hormone`, `follicle_stimulating_hormone`, `lh_fsh_ratio`, `luteinizing_hormone` |
| ovarian_morphology | 9 | `estimated_follicle_number_per_ovary`, `follicle_count_left`, `follicle_count_right`, `follicle_number_per_ovary`, `follicle_number_per_section`, `large_or_uncertain_cystic_structure`, `ovarian_morphology_evidence`, `ovary_area_mm2`, `ovary_volume_ml` |
| reproductive | 18 | `amenorrhea`, `cycle_irregularity`, `cycle_length`, `cycle_phase`, `cycle_regularity`, `e3g`, `estradiol`, `follicle_stimulating_hormone`, `infertility_history`, `lh_fsh_ratio`, `luteinizing_hormone`, `menses_duration`, `menstrual_flow`, `menstrual_frequency_per_year`, `pdg`, `pregnancy_history_count`, `progesterone`, `urinary_lh` |
| symptom | 4 | `fatigue`, `mood_change`, `pelvic_pain`, `weight_gain` |
| wearable | 5 | `activity_steps`, `hrv_rmssd`, `resting_heart_rate`, `skin_temperature`, `sleep_duration_hours` |
<!-- AUTO-GENERATED: VARIABLE-REGISTRY END -->

## Phenotype domains

<!-- AUTO-GENERATED: PHENOTYPE-DOMAINS START -->
| Domain | Features | Min coverage to report | Qualifier |
|:--|--:|--:|:--|
| **reproductive** — Reproductive / ovulatory | 9 | 0.34 | reported menstrual-pattern evidence |
| **metabolic** — Metabolic | 12 | 0.34 | reported metabolic-symptom evidence |
| **clinical_androgenic_evidence** — Clinical androgenic evidence | 5 | 0.25 | — |
| **biochemical_androgenic_evidence** — Biochemical androgenic evidence | 4 | 0.25 | — |
| **ovarian** — Ovarian / LH-AMH | 7 | 0.25 | — |
| **lh_amh_pattern** — LH-AMH reproductive pattern | 5 | 0.34 | — |
| **symptom_burden** — Symptom burden | 8 | 0.34 | patient-reported symptom burden |
<!-- AUTO-GENERATED: PHENOTYPE-DOMAINS END -->

## Processing artifacts

Every ingestion run writes:

```text
artifacts/manifests/<dataset>/<version>/
├── raw_manifest.json        ├── validation_report.json
├── file_checksums.json      ├── dropped_records.csv
├── variable_mapping.json    ├── processing_config.yaml
└── processed_manifest.json
```

The same raw input and config produce the same processed manifest. Invalid
values are reported in `dropped_records.csv`, never silently corrected.

## Adding a dataset

Open a dataset-request issue. `allowed_uses` and `prohibited_claims` are decided
**before** an adapter is written — deciding what a dataset may claim after seeing
results is how unsupported claims happen.
