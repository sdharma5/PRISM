# PRISM

**Personalized Reproductive and Integrated Systemic Model**

A modular, condition-agnostic hormonal-health pipeline that converts several
forms of patient information into standardized, traceable evidence, then trains
*separate* reusable representations for stable clinical phenotype, time-varying
hormonal state, ovarian morphology, confirmed symptom events, and parsed
document events.

<div class="prism-safety">
<strong>PRISM does not diagnose any condition.</strong> It is a research
artifact producing research phenotype profiles, evidence summaries, morphology
measurements, model-estimated probabilities, missing-information analyses and
stability analyses — never confirmed diagnoses, medical advice, treatment
recommendations, guaranteed subtypes, or validated clinical decision support.
</div>

## The constraint that shapes everything

> Do not pretend that unrelated datasets belong to the same patients.

The datasets PRISM ingests describe **different people**. They train separate
modules and are never combined into artificial multimodal patients.
See [ADR-002](decisions/ADR-002-no-fake-pairing.md).

## Start here

- [Quick start](quickstart.md) — install and run in four commands
- [Event store](concepts/event_store.md) — the substrate everything else sits on
- [Provenance](concepts/provenance.md) — what every value carries
- [Missingness](concepts/missingness.md) — six kinds of absent
- [Trait vs state](concepts/trait_vs_state.md) — the two things PRISM models
- [Phenotype profiles](concepts/phenotype_profiles.md) — profiles, not subtypes

## Implementation status

<!-- AUTO-GENERATED: IMPLEMENTATION-STATUS START -->
| Step | Component | Status | Independently validated? |
|:--|:--|:--|:--|
| 1 | Schemas and registries | Implemented | Contract tests |
| 2 | Ingestion + event store | Implemented | Unit tests, synthetic fixtures |
| 3 | Static baselines | Implemented | Cross-validated on the public cohort |
| 4 | Phenotype domains | Implemented | Reconstruction vs mean-imputation baseline |
| 5 | Subtype + stability | Implemented | Stability metrics only; no external validation |
| 6 | Speech pipeline | Implemented | Synthetic scripted corpus only |
| 7 | Document pipeline | Implemented | Synthetic report corpus only |
| 8 | Ultrasound pipeline | Implemented | Segmentation/counting metrics on labelled data |
| 9 | Dynamic hormonal state | Implemented | Grouped participant-level held-out evaluation |
| — | Cross-modal fusion | **Not implemented** | Requires genuinely matched multimodal patients |
<!-- AUTO-GENERATED: IMPLEMENTATION-STATUS END -->

"Independently validated" means evaluated on held-out data *from its own
dataset*. No cross-module or clinical validation is claimed.
