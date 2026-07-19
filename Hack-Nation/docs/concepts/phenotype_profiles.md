# Phenotype profiles

PRISM does not assign a subtype. It reports how a participant's evidence is
organized across domains, which described profile that pattern most
**resembles**, how confident and how stable that resemblance is, and when it
should decline to answer. Language rules are fixed by
[ADR-003](../decisions/ADR-003-subtype-language.md).

The continuous **domain scores are the primary phenotype output**. The profile
similarities are **secondary and exploratory** — comparisons to patterns
described in the literature, with no ground-truth subtype anywhere in this
repository to be right or wrong against. Nothing here is a validated clinical
subtype.

## Two representations, deliberately both

**A. Transparent composite scores.** Deterministic, auditable, driven entirely
by `registry/phenotype_domains.yaml`:

$$ s_d = \frac{\sum_{j \in d} w_j z_j m_j}{\sum_{j \in d} w_j m_j} $$

where \(z_j\) is the standardized feature, \(w_j\) a documented weight, and
\(m_j\) an availability indicator. Every score reports its coverage.

**B. Learned tabular embedding.** A masked autoencoder over clinical features +
missingness mask + variable identities, trained by masking 10–30% of observed
variables and reconstructing them. Exported as a latent embedding, and required
to beat mean imputation on withheld values — otherwise it is adding nothing.

Both are exported. The composite score is what a clinician can argue with; the
embedding is what clusters well. Neither is asked to be the other.

## The domains

`registry/phenotype_domains.yaml` v1.2.0. Each domain declares an
`evidence_source`, so what kind of evidence a score rests on is readable off
the registry rather than inferred from the feature list.

| Domain | Captures | `evidence_source` |
|:--|:--|:--|
| Reproductive / ovulatory | Cycle pattern, LH/FSH, progesterone | mixed |
| Metabolic | Adiposity, glycemia, lipids, blood pressure, acanthosis nigricans | mixed |
| Clinical androgenic evidence | Ferriman–Gallwey, hirsutism, acne, alopecia, facial hair | symptoms |
| Biochemical androgenic evidence | Total/free testosterone, DHEAS, SHBG | biochemical |
| Ovarian / LH–AMH | AMH, follicle count, ovarian volume | imaging |
| LH–AMH reproductive pattern | LH, LH/FSH, AMH against comparatively unremarkable adiposity | mixed |
| Symptom burden | Patient-reported symptom load | symptoms |

**Why androgenic evidence is two domains, not one.** Merged, the weight of
un-drawn assays sat in the coverage denominator: a patient with recorded
hirsutism and acne but no androgen panel fell below the coverage floor, and the
entire androgenic axis abstained even though the cutaneous signs had been
observed. Scored separately, each domain has its own denominator. In the PMOS
tabular cohort clinical androgenic coverage is 1.5/3.5 = 0.43 — **assessable
for all 541 patients** — and biochemical coverage is 0/4.5, **unavailable for
all 541**, because that cohort contains no androgen assay at all.

**The androgenic caveat is not decoration.** Every output states an
`androgenic_evidence_source`: `symptoms_only`, `biochemical_only`, `both`, or
`unavailable`. In this cohort it is always `symptoms_only`. Symptoms are not
biochemical hyperandrogenism — the correlation is real but far from identity,
and it varies by ancestry in ways that make the substitution actively unfair.
A clinical androgenic score is never reported, or read, as a measured androgen
level.

`skin_darkening` sits in `metabolic`, not with the androgenic signs: acanthosis
nigricans is a sign of insulin resistance, not androgen excess.

## Coverage, and refusing to score

```json
{
  "domain": "metabolic",
  "score": 0.72,
  "coverage": 0.60,
  "observed_features": ["BMI", "fasting_glucose", "blood_pressure"],
  "missing_features": ["fasting_insulin", "lipids"]
}
```

Below the domain's `min_coverage_to_report`, `score` is `None`. A score computed
from one of nine features is not a weak score — it is a different quantity, and
reporting it as 0.72 invites it to be read as comparable to a well-covered one.

## Eligibility comes before scoring

A profile is scored only if the domain that *defines* it was actually
assessable for that participant:

| Profile | Requires |
|:--|:--|
| `metabolic_leaning` | `metabolic` assessable |
| `lh_amh_leaning` | `lh_amh_pattern` assessable |
| `androgenic_leaning` | `clinical_androgenic_evidence` **or** `biochemical_androgenic_evidence` assessable |
| `mixed` | at least two assessable domains |

Ineligible profiles are **removed from the catalogue** before any similarity is
computed, and the remaining similarities are renormalized over what is left.
They are never zero-filled, and never scored and then suppressed — a
renormalized similarity over three eligible profiles is an honest quantity; a
similarity over four with one silently set to zero is not. If nothing is
eligible, the result is `indeterminate`.

## Soft membership with an escape hatch

```json
{
  "phenotype_probabilities": {
    "metabolic_leaning": 0.55,
    "lh_amh_leaning": 0.25,
    "androgenic_leaning": 0.12,
    "indeterminate": 0.08
  },
  "androgenic_evidence_source": "symptoms_only"
}
```

`indeterminate` is assigned when no profile is eligible, the maximum
probability is below threshold, the top two profiles are near-tied, bootstrap
assignment is unstable, removing a single variable flips the dominant profile,
the participant sits far from every prototype, or too few defining domains were
observed.

## Stability travels with the answer

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

Read that as: *the profile resembles a metabolic-leaning pattern, but one in five
resamples assigns it elsewhere, and fasting insulin is doing most of the work.*
That is a more useful sentence than a subtype label, and it is the one the data
supports.

`dominant_profile` is populated **only** when the stability engine classifies
the assignment as stable. When it does not, the field is `null`,
`indeterminate` is `true`, and a reason is attached:

```json
{
  "dominant_profile": null,
  "indeterminate": true,
  "indeterminate_reasons": ["assignment unstable under resampling"]
}
```

An unstable assignment is returned as indeterminate rather than as a labelled
profile carrying a low stability score, because a name next to a caveat gets
read as the name.
