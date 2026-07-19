# Dataset registry

No clinical data is committed to this repository. The machine-readable source of
truth is `registry/datasets.yaml`; adapters call
`DatasetRegistry.require(dataset_id, use)` and **raise `PermissionError`** on a
prohibited use.

<!-- AUTO-GENERATED: DATASET-REGISTRY START -->
| Dataset | Access | Longitudinal | Allowed uses | Prohibited claims |
|:--|:--|:--|:--|:--|
| **mcphases**<br><sub>mcPHASES</sub> | restricted_physionet | Yes | `temporal_state_model`<br>`cycle_phase_prediction`<br>`hormone_reconstruction`<br>`symptom_forecasting`<br>`missingness_analysis` | `pcos_subtype_validation`<br>`pcos_diagnosis`<br>`binary_baseline` |
| **nhanes_2021_2023**<br><sub>NHANES 2021-2023</sub> | public_cdc_download | No | `population_reference`<br>`unit_harmonization`<br>`external_stress_testing`<br>`missingness_analysis` | `longitudinal_state_modeling`<br>`validated_pcos_subtypes`<br>`pcos_diagnosis` |
| **ovarian_ultrasound_2d**<br><sub>2D ovarian ultrasound dataset (to be selected)</sub> | external_download_pending_selection | No | `ovary_classification`<br>`image_quality_modeling`<br>`ultrasound_embedding`<br>`pcom_morphology_classification`<br>`external_stress_testing` | `pcos_diagnosis`<br>`validated_pcos_subtypes`<br>`follicle_instance_segmentation`<br>`prospective_clinical_deployment` |
| **pcos_tabular_public**<br><sub>Public PCOS clinical tabular dataset</sub> | public_or_external_download | No | `binary_baseline`<br>`phenotype_domain_modeling`<br>`exploratory_clustering`<br>`missingness_analysis` | `validated_four_subtype_classification`<br>`prospective_clinical_deployment`<br>`longitudinal_state_modeling` |
| **prism_document_eval_synthetic**<br><sub>PRISM synthetic laboratory-report evaluation set</sub> | in_repository_synthetic | No | `document_extraction_evaluation`<br>`unit_conversion_evaluation`<br>`page_grounding_evaluation` | `clinical_document_validation`<br>`real_world_performance_estimate` |
| **prism_speech_eval_synthetic**<br><sub>PRISM scripted speech evaluation set (synthetic)</sub> | in_repository_synthetic | No | `speech_extraction_evaluation`<br>`negation_evaluation`<br>`temporality_evaluation` | `clinical_speech_validation`<br>`real_world_performance_estimate` |
| **usova3d**<br><sub>USOVA3D</sub> | public_or_external_download | No | `ovary_segmentation`<br>`follicle_segmentation`<br>`follicle_counting`<br>`morphology_measurement`<br>`slice_extraction_for_2d_pretraining`<br>`optional_3d_benchmark` | `pcos_diagnosis`<br>`validated_pcos_subtypes`<br>`prospective_clinical_deployment`<br>`independent_2d_evaluation` |
<!-- AUTO-GENERATED: DATASET-REGISTRY END -->

## Access

| Dataset | How to obtain | Note |
|:--|:--|:--|
| Public PCOS tabular | External download | Ultrasound columns are report-transcribed, not image-measured |
| mcPHASES | Credentialed PhysioNet access + DUA | The only genuinely longitudinal dataset here |
| NHANES 2021–2023 | Public CDC download | Complex survey design; weights required for population estimates |
| USOVA3D | External download | Annotator-derived masks; metrics inherit annotator variability |
| PRISM speech / document sets | Committed, synthetic | Support no real-world performance claim |

Store real data outside the repository under `PRISM_DATA_ROOT`.

## Processing artifacts

```text
artifacts/manifests/<dataset>/<version>/
├── raw_manifest.json        ├── validation_report.json
├── file_checksums.json      ├── dropped_records.csv
├── variable_mapping.json    ├── processing_config.yaml
└── processed_manifest.json
```

Same raw input + same config → same processed manifest. Invalid values are
reported in `dropped_records.csv`, never silently corrected.
