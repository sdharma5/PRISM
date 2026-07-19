# Provenance

<div class="prism-safety">
Every structured value in PRISM can answer: where did you come from, when, in
what unit, who or what produced you, how confident were they, has a human
checked you, and which model version made you?
</div>

## What every value carries

`HormonalHealthEvent` is the unit of evidence. Alongside the value it records:

| Field | Question it answers |
|:--|:--|
| `source_dataset`, `source_file_id`, `source_file_hash` | Which artifact? |
| `observed_at` / `start_at` / `end_at` | When was it true? |
| `unit` + `raw_unit` | In what unit — and what did the source actually say? |
| `value` + `raw_value` | Canonical value, and the untouched original |
| `modality` | What kind of channel produced it? |
| `provenance` | Measured, dataset-provided, extracted, or inferred? |
| `extraction_confidence` | How sure was the producer? |
| `confirmation_status`, `reviewed_by`, `reviewed_at` | Has a human signed off? |
| `evidence_text`, `source_page`, `source_time_*_seconds` | Where exactly? |
| `parser_version`, `model_version`, `schema_version` | Which code produced it? |

`raw_value` and `raw_unit` are the ones people forget. Normalization is a
model of the world and models are wrong; keeping the original means a unit-table
bug is recoverable rather than baked into every derived artifact.

## The provenance ladder

Ordered by trust, and used by snapshot selection when two events disagree:

```text
clinician_confirmed   ← a clinician asserted it
patient_confirmed     ← the patient asserted it
device_measured       ← an instrument recorded it
dataset_provided      ← a curated dataset column
document_extracted    ← a parser read it off a page      ⚠ needs review
model_measured        ← a segmentation model measured it ⚠ needs review
model_inferred        ← a model guessed it               ⚠ needs review
```

The three marked entries can never be `confirmed` without a `reviewed_by`
value — `schemas/event.py` raises otherwise. See
[ADR-004](../decisions/ADR-004-human-confirmation.md).

## Evidence spans are mandatory for unstructured sources

For `patient_voice`, `clinician_voice`, `clinical_document`, and
`ultrasound_report`, an observed event **must** carry `evidence_text`, a
`source_page`, or a time offset. The schema enforces it.

This is what makes review possible at all. A reviewer confirming "cycle
irregularity: present" needs to see *"my periods have been between 45 and 70 days
apart for about a year"* and the seconds it occupies in the recording. Without
the span, confirmation is rubber-stamping.

It also gives the document pipeline a hard failure mode: a lab value that cannot
be grounded to a page span is **dropped and counted as unsupported**, never
silently added. `unsupported-value rate` is a reported metric precisely because
a confidently hallucinated potassium level is worse than a missing one.

## Provenance survives to the result

Because the store is append-only and snapshots record their selections, every
number in a metric table traces back:

```text
aggregate AUROC
  └─ predictions.parquet
      └─ feature_manifest.json      which columns, which transforms
          └─ PatientSnapshot        which event won, how many candidates, what was excluded
              └─ HormonalHealthEvent
                  └─ page 2, "Testosterone, Total   62 ng/dL   (8-60)"
```

`event_store/provenance.py` walks that chain for any event id.
