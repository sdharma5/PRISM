# Experiments

Every experiment is config-driven, seeded, and reproducible from a git commit
plus a configuration file.

## Artifact contract

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

Each experiment README states its goal, dataset, target, split method, model,
metrics, limitations, and reproduction command.

## Configured experiments

| Config | What it answers |
|:--|:--|
| `exp_static_baselines.yaml` | Does the pipeline and evaluation methodology work at all? |
| `exp_phenotype_domains.yaml` | Do learned embeddings beat mean imputation on withheld variables? |
| `exp_subtype_stability.yaml` | Are discovered profiles stable under resampling and ablation? |
| `exp_speech_extraction.yaml` | How accurately are spoken symptoms extracted and grounded? |
| `exp_document_extraction.yaml` | Are lab values, units and pages extracted correctly? |
| `exp_ultrasound.yaml` | Ovary/follicle segmentation, counting, and volume error |
| `exp_dynamic_state.yaml` | Hormone reconstruction and cycle-state prediction on held-out participants |

## Recorded results

<!-- AUTO-GENERATED: EXPERIMENTS START -->
| Experiment | Model | Dataset version | Primary metrics |
|:--|:--|:--|:--|
| `exp_dynamic_state` | longitudinal_hormonal_state_model | unversioned | accuracy=0.438, balanced_accuracy=0.434, bloating_auprc=0.337 |
| `exp_phenotype_domains` | tabular_masked_autoencoder | kottarathil-2020 | masked_reconstruction_mse=0.756, mean_imputation_mse=1.019, mse_improvement_over_mean=0.263 |
| `exp_static_baselines:static_logistic` | static_logistic | kottarathil-2020 | n=108.200, prevalence=0.327, auroc=0.943 |
| `exp_usova3d_2d_unet` | — | — | — |
| `exp_usova3d_2d_unet_v2` | — | — | — |
| `exp_usova3d_3d_unet` | — | — | — |
<!-- AUTO-GENERATED: EXPERIMENTS END -->

## Rules that make results comparable

1. The primary metric is declared **before** the run, in the issue template.
2. Preprocessing is fitted inside the training fold.
3. Splits are patient-level; longitudinal splits are grouped by participant.
4. At least five seeds for small-sample work; report mean and standard deviation.
5. Calibration is always reported, not just discrimination.
6. Smoke-test numbers are never quoted as results.
