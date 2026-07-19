# Model Card — PMOS Evidence Profile and Phenotype Similarity

**Version:** 0.1.0 · **Status:** research only · **Clinician confirmation required**

## What this produces

For one patient, from whatever modalities they supply:

* a **PMOS evidence probability** (learned),
* **Rotterdam-axis evidence** (rule-based, published thresholds),
* **continuous phenotype-domain scores** across seven domains — the primary
  phenotype output,
* **phenotype-profile similarities** (rule-based, prototype comparison) — a
  secondary, exploratory output,
* abstention, missing evidence, conflicts, and a per-section explanation.

## The single most important distinction

| Component | Learned? | Fit against what |
|:--|:--|:--|
| PMOS evidence probability | **Yes** | `pmos_binary` on 432 training patients, one clinic, cross-sectional |
| Phenotype domain scores | No | Documented weighted mean of z-scores (`registry/phenotype_domains.yaml`) |
| Phenotype profile similarity | No | Cosine to **declared** centroids from the literature |
| Rotterdam axes | No | Published 2023 Guideline thresholds |
| Cross-modal coordination | No | Declared design-rule weights |

Every output carries `learned_components_used` and `rule_based_components_used`
so this table can be reconstructed from any individual result.

**Only the static clinical head was trained on a PMOS label.** Held-out AUROC
0.8927, AUPRC 0.8536, raw Brier 0.1270, calibrated Brier 0.1147 on 109 patients
unseen during fitting (432 training patients).

## How the probability is calibrated

A `PlattCalibrator` is fitted on **out-of-fold predictions from the 432
training patients only** — it refuses any input not marked
`source="train_out_of_fold"` — and is then applied **once, frozen**, to the 109
held-out patients (coefficient 0.814, intercept −0.649). Nothing is ever fitted
on held-out labels. Both `raw_model_score` and `calibrated_model_score` are
preserved in every output, so the effect of calibration is always visible.

Reliability is reported over **five equal-frequency bins**. Each bin carries its
patient count and a 95% Wilson binomial confidence interval, plus an
`interpretable` flag that is `false` below 20 patients in the bin. That flag is
the point: **calibration is uncertain wherever the score range is sparsely
represented.** On 109 held-out patients the extreme bins are thin, their
intervals are wide, and a calibrated probability from those ranges should not be
read as a well-estimated one. Aggregate Brier improvement does not license
trusting any individual sparsely-supported score.

## Phenotype profiles are NOT validated subtypes

The profiles — `metabolic_leaning`, `lh_amh_leaning`, `androgenic_leaning`,
`mixed`, `indeterminate` — are **exploratory similarities to described research
patterns**, not clinical subtypes. The continuous domain scores are the primary
phenotype output; the similarities are secondary.

* **No subtype label exists anywhere in this repository.** Nothing was fit to,
  or validated against, a ground-truth subtype.
* The centroids are **declared from the literature**, not learned. Fitting them
  on 541 unlabelled patients would produce cluster artifacts wearing literature
  names; that is worse than declaring them, because it would look validated.
* Language is enforced in code: `models/phenotype/prototype_mapping.py` raises
  `ProhibitedLanguageError` on unhedged descriptions.

Correct phrasing is "resembles the metabolic-leaning research profile." Incorrect
is "has the metabolic subtype."

### Eligibility, not zero-filling

A profile is scored only when the domain that defines it was assessable:
`metabolic_leaning` requires `metabolic`; `lh_amh_leaning` requires
`lh_amh_pattern`; `androgenic_leaning` requires `clinical_androgenic_evidence`
or `biochemical_androgenic_evidence`; `mixed` requires at least two assessable
domains. Ineligible profiles are **removed before scoring** and the remaining
similarities renormalized — never zero-filled, never scored and then
suppressed. No patient can be called `androgenic_leaning` without androgenic
input.

### Unstable assignments are returned as indeterminate

`dominant_profile` is populated **only** when the stability engine classifies
the assignment as stable. Otherwise `dominant_profile` is `null`,
`indeterminate` is `true`, and a reason is given. An `indeterminate` outcome is
a first-class result, reachable several ways: no eligible profile, thin
evidence, near-tie between top profiles, best similarity below the floor, or an
unstable assignment. On real cohort patients these fire routinely.

## The seven domains

`reproductive`, `metabolic`, `clinical_androgenic_evidence`,
`biochemical_androgenic_evidence`, `ovarian`, `lh_amh_pattern`,
`symptom_burden` (`registry/phenotype_domains.yaml` v1.2.0). Each declares an
`evidence_source` of `symptoms`, `biochemical`, `imaging` or `mixed`.

A domain returns **None**, never 0.0, when its coverage falls below the declared
floor. This matters: a z-score of 0.0 means "exactly average", which is a
measurement; `None` means "not assessed".

**The androgenic domain was split at v1.2.0.** Merged, the weights of un-drawn
assays sat in the coverage denominator, so observed cutaneous signs fell under
the 0.25 floor and the whole androgenic axis abstained for all 541 patients.
Split, `clinical_androgenic_evidence` (report-class signs) has coverage
1.5/3.5 = 0.43 and is **assessable for all 541**, while
`biochemical_androgenic_evidence` (assays only) has coverage 0/4.5 and is
**unavailable for all 541** — this cohort carries no androgen assay at all.
Every output states an `androgenic_evidence_source`: `symptoms_only`,
`biochemical_only`, `both`, or `unavailable`. **In this cohort it is always
`symptoms_only`**, and a clinical androgenic score is not a measured androgen
level.

`skin_darkening` moved to `metabolic` at v1.2.0 — acanthosis nigricans is a sign
of insulin resistance, not androgen excess.

`symptom_burden` is entirely report evidence and is never sufficient for a
diagnostic axis on its own.

## Modality roles

* **Static clinical** — the learned PMOS probability and all domain scores.
* **Ultrasound** — ovarian-morphology evidence only. Cannot produce a PMOS
  probability; an ultrasound-only patient abstains from the whole-patient claim
  while still receiving morphology findings.
* **Temporal** — current-state and recent ovulatory evidence only. Cannot produce
  a PMOS probability, subtype, or Rotterdam determination.
* **Speech / documents** — ingestion. They populate confirmed clinical variables
  consumed by the static encoder; they are never independent diagnostic branches.

## No cross-modal relationship was learned

Per `docs/decisions/ADR-002-no-fake-pairing.md`, the datasets describe different
people. No fusion model is trained, `joint_model_used=True` is unreachable by
schema validation, and combining across modalities uses declared weights whose
values were chosen, not fitted. **No accuracy figure applies to the combined
multimodal result** — only to the static branch in isolation.

## Known limitations

1. **Single-cohort learned component.** 541 patients, one clinic, cross-sectional.
   Calibration outside that population is unknown; no external validation.
   Within the cohort, calibration is **uncertain in sparsely represented score
   ranges** — reliability bins below 20 patients are flagged
   `interpretable: false` and their Wilson intervals are wide.
2. **No androgen assays** in the cohort, so `biochemical_androgenic_evidence` is
   `None` for every patient and the biochemical hyperandrogenism axis is always
   `not_assessable`. Androgenic evidence here is **symptom-based only**, and is
   labelled `symptoms_only` on every output.
3. **Biochemical androgen thresholds are placeholders** and assay-dependent.
4. **Prototype centroids are declared, not fitted**, the profile similarity has
   no ground truth to be right or wrong against, and the similarities are
   exploratory rather than validated.
5. **Ultrasound evidence currently comes from a weak model** — held-out follicle
   Dice 0.49, instance F1 0.05 on 2 test volumes. Morphology evidence should be
   treated as provisional until the retrained model is evaluated.
6. **Temporal evidence is short-horizon.** A 14-day window cannot characterise
   long-term cycle regularity; the coordinator raises this explicitly when static
   history and temporal state disagree.
7. **A stability verdict is required, not optional.** `dominant_profile` is
   populated only for assignments the stability engine calls stable; an
   unstable assignment, or one that was never checked, is returned as
   `indeterminate` with a reason. "Not checked" is not evidence of stability.

## Prohibited uses

* Any diagnostic use, including polycystic ovarian morphology determination.
* Describing a profile as a clinical subtype or a diagnosis.
* Reporting the phenotype similarity as validated; it is exploratory.
* Reading a clinical androgenic score as biochemical hyperandrogenism.
* Quoting a calibrated probability from a score range whose reliability bin is
  flagged `interpretable: false`.
* Using an ultrasound-only or temporal-only result as a PMOS assessment.
* Quoting a combined multimodal accuracy figure — none exists.
