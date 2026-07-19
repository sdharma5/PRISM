# Subtype and stability engine (Step 5)

Discovers reproducible phenotype patterns **without** forcing every person into
a rigid subtype. Language rules: [ADR-003](../decisions/ADR-003-subtype-language.md).

## Population

Clustering runs only on participants with a positive PCOS label in the
**training** data. The function requires an explicit subset — it will not
silently cluster everyone.

## What is compared

**Representations:** raw standardized features · transparent domain scores ·
autoencoder embeddings · published-variable subsets.

**Algorithms:** k-means · Gaussian mixture · hierarchical (Ward) · spectral
(optional) · consensus clustering.

**K:** 2, 3, 4, 5, 6. Four is never chosen merely because publications report
four groups.

## Stability analysis

Each configuration is re-run across multiple seeds, bootstrap resamples, feature
subsets, alternative scaling, alternative imputation, missing-modality
ablations, and plausible laboratory perturbations drawn from documented assay
coefficients of variation.

**Metrics:** silhouette · Calinski-Harabasz · Davies-Bouldin · ARI · NMI ·
bootstrap Jaccard · assignment entropy · subtype flip rate · Jensen-Shannon
divergence.

## Indeterminate

Assigned when the maximum probability is below threshold, models disagree,
bootstrap assignment is unstable, removing one variable flips the dominant
cluster, the participant lies far from all cluster centers, or too few defining
variables were observed.

## Output

```json
{
  "dominant_profile": "metabolic_leaning",
  "dominant_probability": 0.61,
  "stability_score": 0.73,
  "subtype_flip_rate": 0.21,
  "highest_fragility_feature": "fasting_insulin",
  "abstain": false,
  "warnings": ["SHBG unavailable", "Only one glucose measurement available"]
}
```

Descriptions may only say *resembles*, *is most similar to*, *has overlap with*.
A banned-phrase guard in `models/phenotype/prototype_mapping.py` raises on
"clinically validated subtype" and friends, and a unit test asserts it fires.
