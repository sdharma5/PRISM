# ADR-004: Human confirmation gates uncertain evidence

**Status:** Accepted · **Date:** 2026-07-18

## Context

Three of PRISM's inputs are produced by models, not measured:

- speech extraction turns a sentence into a structured symptom event;
- document extraction turns a PDF cell into a lab value;
- ultrasound segmentation turns voxels into a follicle count.

Each is useful and each is wrong sometimes. A transcription error, a
misparsed reference range, or an over-segmented follicle cluster is
indistinguishable from a real observation once it lands in a feature matrix —
the value looks the same. The damage is silent and downstream.

## Decision

Confirmation status is part of the event contract, not a workflow afterthought.

| Source | Provenance | Default confirmation status |
|:--|:--|:--|
| Dataset column | `dataset_provided` | `not_required` |
| Device stream | `device_measured` | `not_required` |
| Speech extraction | `patient_voice` / `clinician_voice` | `awaiting_patient_confirmation` |
| Document extraction | `document_extracted` | `awaiting_clinician_confirmation` |
| Ultrasound model | `model_measured` | `awaiting_clinician_confirmation` |
| Any inference | `model_inferred` | never `confirmed` on its own |

Enforced mechanically in `schemas/event.py`:

- an event whose provenance is `document_extracted`, `model_measured` or
  `model_inferred` **cannot** be `confirmed` without a `reviewed_by` value;
- `is_model_ready` is true only for `confirmed` or `not_required` events, and
  `build_snapshot(...)` filters on it;
- speech, document and ultrasound-report events **must** carry an evidence span
  or source location, so a reviewer can always see what they are confirming.

Imaging carries a second axis: `OvarianMorphologyOutput.clinician_review_status`
starts at `model_generated` and only a clinician moves it to
`clinician_confirmed`. The exported ultrasound token warns
"Clinician confirmation pending" until then.

## Consequences

**Cost.** A review step sits between extraction and modeling, and unreviewed
extractions are worthless to the model even though they are already computed.

**Benefit.** No model-generated number can impersonate a measurement. The
distinction survives serialization, snapshotting, and export, which means a
downstream consumer who never read this document still cannot confuse the two.

**Corollary for evaluation.** Extraction quality is measured against the
*pre-confirmation* output (that is what the model actually produced), while
modeling uses only *post-confirmation* events. Conflating the two would flatter
the extractor and contaminate the model.
