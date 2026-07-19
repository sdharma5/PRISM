# Phenotype domains (Step 4)

Moves beyond a single binary prediction to continuous, domain-level
representations. Concepts and the androgenic caveat are in
[phenotype profiles](../concepts/phenotype_profiles.md); this page is the
implementation.

These continuous domain scores are the **primary** phenotype output. The soft
profile similarities built on top of them are secondary and exploratory, and
are not validated clinical subtypes.

## A. Transparent composite scores

Deterministic, registry-driven, auditable:

$$ s_d = \frac{\sum_{j \in d} w_j z_j m_j}{\sum_{j \in d} w_j m_j} $$

Weights, directions and evidence classes live in
`registry/phenotype_domains.yaml` (v1.2.0) — changing the scoring means editing
a reviewed config file, not hunting for a constant in Python. Each domain also
declares an `evidence_source`: `symptoms`, `biochemical`, `imaging` or `mixed`.

### The androgenic split

`androgenic` is no longer one domain. It is now
`clinical_androgenic_evidence` (report-class only: `ferriman_gallwey_score`,
`hirsutism`, `acne`, `androgenic_alopecia`, `hair_growth_face`) and
`biochemical_androgenic_evidence` (assay-only: `total_testosterone`,
`free_testosterone`, `dheas`, `shbg`).

The merged version put the weights of assays that were never drawn into the
coverage denominator. Observed cutaneous signs were therefore dragged below the
0.25 floor and the androgenic axis abstained for every patient in the cohort —
discarding evidence that had in fact been recorded. Scored on separate
denominators, clinical coverage is 1.5/3.5 = 0.43 and is assessable for all 541
patients; biochemical coverage is 0/4.5 and is unavailable for all 541, because
the cohort holds no androgen assay.

Consumers receive an explicit `androgenic_evidence_source` — `symptoms_only`,
`biochemical_only`, `both`, or `unavailable`. It is always stated. On this
cohort it is always `symptoms_only`, and a clinical androgenic score must never
be reported as a measured androgen level.

`skin_darkening` moved to the `metabolic` domain: acanthosis nigricans is a
sign of insulin resistance, not androgen excess.

Every output reports coverage:

```json
{
  "domain": "metabolic",
  "score": 0.72,
  "coverage": 0.60,
  "observed_features": ["BMI", "fasting_glucose", "blood_pressure"],
  "missing_features": ["fasting_insulin", "lipids"]
}
```

Below `min_coverage_to_report`, `score` is `None`. Refusing to score is a valid
output.

## B. Learned tabular representation

A masked/denoising autoencoder over clinical features + missingness mask +
variable identities. 10–30% of observed variables are randomly masked and
reconstructed:

$$ \mathcal{L}_{MAE} = \sum_{j \in M_c} \alpha_j (x_j-\hat{x}_j)^2 + \sum_{j \in M_k} \beta_j \operatorname{CE}(x_j,\hat{x}_j) $$

The acceptance bar is explicit: the embedding must reconstruct withheld
variables **better than mean imputation**. If it does not, it is adding
parameters and nothing else.

## Exported token

```json
{
  "modality": "static_clinical",
  "embedding": [0.12, -0.31, 0.44],
  "structured_features": {
    "metabolic_score": 0.72, "reproductive_score": 0.81,
    "clinical_androgenic_evidence_score": 0.54,
    "biochemical_androgenic_evidence_score": null,
    "androgenic_evidence_source": "symptoms_only",
    "ovarian_score": 0.62
  },
  "quality_score": 0.78,
  "confidence_score": 0.69,
  "missing_fields": ["fasting_insulin", "SHBG"]
}
```

Scores and embeddings are exported **separately**. They answer different
questions and combining them would hide which one is doing the work.
