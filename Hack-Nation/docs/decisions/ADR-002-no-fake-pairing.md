# ADR-002: Never fabricate multimodal patients

**Status:** Accepted · **Date:** 2026-07-18 · **Supersedes:** none

## Context

PRISM ingests five modalities. The obvious next move — and the one most
multimodal papers make — is to concatenate the per-modality tokens and train a
fusion network. We cannot do that here, because the datasets describe different
people:

| Modality | Dataset | Population |
|:--|:--|:--|
| Static clinical | Public PMOS tabular cohort | One clinic, cross-sectional |
| Longitudinal state | mcPHASES | Credentialed cohort, densely sampled |
| Population reference | NHANES 2021–2023 | US survey sample |
| Ultrasound | USOVA3D | Imaging cohort with annotator masks |
| Speech / documents | PRISM synthetic sets | Not real people at all |

Pairing a random tabular row with a random ultrasound volume and calling the
result "patient 47" manufactures a correlation structure that does not exist.
Any fusion model trained that way learns dataset artifacts, and its accuracy is
an artifact too — but it will *look* like a multimodal result, which is worse
than looking like nothing.

## Decision

1. `patient_id` is scoped by `source_dataset`. Identifiers are never merged
   across datasets, and `PatientRecord` carries the dataset it came from.
2. Each modality trains an independent encoder that exports a `ModalityToken`.
   The tokens share an envelope so they are *comparable*, not so they are
   *concatenable*.
3. No joint fusion model is trained in Steps 1–9. A fusion model becomes
   legitimate only when genuinely matched multimodal patient data exists.
4. A synthetic fictional case may demonstrate the user interface. Synthetic
   combinations must never appear in any scientific evaluation, metric table, or
   model card.
5. Documentation states, per module, which dataset it was validated on — and
   that no cross-module validation exists.

## Consequences

**Accepted cost.** We give up the headline multimodal number. Step 9 ends with
five separately validated modules rather than one impressive-looking model.

**Benefit.** Every number in the repository refers to a real population. When
matched data arrives, the encoders are already trained and the fusion layer is
the only new thing to validate — the honest version is also the reusable one.

**Enforcement.** The shared envelope makes accidental fusion easy to write, so
the guard is social and documentary rather than purely mechanical: this ADR,
the README constraint, the per-dataset `prohibited_claims` in
`registry/datasets.yaml`, and the pull-request question *"does this change alter
any medical or scientific claim?"*.
