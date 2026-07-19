# Static baseline model (Step 3)

## Objective

Establish a rigorous static baseline **before** building subtype or multimodal
models. The binary PCOS target exists to verify that the data pipeline and
evaluation methodology work — not because binary classification is the goal.

## Pipeline

```text
Raw participant-level table
        ↓
Patient-level split          ← saved as a split manifest
        ↓
Training-fold-only imputation
        ↓
Training-fold-only scaling
        ↓
Categorical encoding
        ↓
Missingness indicators
        ↓
Model fitting
```

Every preprocessing step is fitted **inside** the fold.
`tests/unit/test_preprocessing_leakage.py` fails if that regresses — leakage
inflates metrics silently and is invisible in a diff.

## Baselines

Logistic regression, random forest, gradient boosting (XGBoost/LightGBM, with an
sklearn fallback), a small MLP, a majority-class baseline, and a simple rule
baseline. A complex model that cannot beat the majority-class baseline has told
you something important.

## Feature groups

Reproductive · hormonal laboratory · metabolic · symptom · anthropometric ·
derived ultrasound · missingness indicators · measurement recency where
available.

## Splits

The dataset is small, so: repeated stratified five-fold cross-validation,
patient-level, with fixed saved split manifests and at least five random seeds.
An untouched final test set is used only if sample size permits.

## Metrics

AUROC · AUPRC · balanced accuracy · sensitivity · specificity · F1 · Brier score
· expected calibration error · calibration slope · calibration intercept.

Calibration is reported alongside discrimination always. A model with an AUROC
of 0.85 and a calibration slope of 0.4 is not usable as a probability, and the
AUROC alone will not tell you that.

## Limitations

Single dataset, cross-sectional, modest sample size, dataset-provided label that
was not re-adjudicated. Results describe this cohort. No external validation
exists.
