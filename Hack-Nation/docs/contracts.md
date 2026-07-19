# Data contracts

Every module boundary in PRISM is typed. A schema change is a scientific change:
bump the version in `registry/schema_versions.yaml`, add a CHANGELOG entry, and
the `data-contracts` workflow will block the PR if either is missing.

## The universal event

`HormonalHealthEvent` is the unit of evidence — see
[provenance](concepts/provenance.md) for the full field list. Its validation
rules are the ones worth knowing:

| Rule | Why |
|:--|:--|
| `observed` requires a value; non-observed forbids one | No half-present cells |
| Laboratory events require a unit | A unitless lab value is uninterpretable |
| `document_extracted` / `model_measured` / `model_inferred` cannot be `confirmed` without `reviewed_by` | Models do not confirm themselves |
| Speech / document / ultrasound-report events require an evidence span or source location | Confirmation without evidence is rubber-stamping |
| `raw_value` and `raw_unit` are preserved | Normalization is a model, and models are wrong |

## The token envelope

All five encoders export the same `ModalityToken` shape:

```json
{
  "patient_id": "P001",
  "modality": "static_clinical",
  "embedding": [],
  "structured_features": {},
  "quality_score": 0.0,
  "confidence_score": 0.0,
  "observed_at": null,
  "model_version": "0.1.0",
  "source_dataset": null,
  "provenance_ids": [],
  "missing_fields": [],
  "warnings": []
}
```

Shared envelope, separate lives — see
[ADR-002](decisions/ADR-002-no-fake-pairing.md).

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

Each domain declares an `evidence_source` — `symptoms`, `biochemical`,
`imaging`, or `mixed` — so a consumer can tell what *kind* of evidence a score
rests on without re-deriving it from the feature list.

**Registry 1.1.0 → 1.2.0: the androgenic split.** The single `androgenic` domain
was separated into `clinical_androgenic_evidence` (report-class signs:
Ferriman–Gallwey score, hirsutism, acne, androgenic alopecia, facial hair
growth) and `biochemical_androgenic_evidence` (assay-only: total and free
testosterone, DHEAS, SHBG). One domain mixing the two put the weight of
un-drawn assays into the coverage denominator, so a patient with observed
cutaneous signs and no androgen panel fell under the coverage floor and the
whole androgenic axis abstained. Split, each is scored on its own denominator:
in the PMOS tabular cohort clinical coverage is 1.5/3.5 = 0.43 and is
**assessable for all 541 patients**, while biochemical coverage is 0/4.5 and is
**unavailable for all 541** — that cohort carries no androgen assay.

`skin_darkening` moved from the androgenic domain to `metabolic`. Acanthosis
nigricans is a sign of insulin resistance; scoring it as androgen excess was a
mis-assignment.

Any output touching hyperandrogenism therefore carries an explicit
`androgenic_evidence_source`, always one of `symptoms_only`,
`biochemical_only`, `both`, or `unavailable`. It is never omitted and never
inferred — for every patient in this cohort it is `symptoms_only`.

## Units

`registry/units.yaml` holds every conversion factor. Molar conversions are
analyte-specific (they depend on molecular weight) and are therefore never
applied generically. `convert_to_canonical()` **raises** on an unknown unit
rather than guessing — a wrong silent conversion is far more damaging than a
loud failure. Every factor has a unit test.
