# The universal hormonal event store

The event store is PRISM's single patient-centric, append-only substrate for
structured evidence. Design rationale lives in
[ADR-001](../decisions/ADR-001-event-store.md); this page is how to use it.

## Responsibilities

- Save events (append-only — nothing is mutated or deleted)
- Query by patient, time window, modality, variable, confirmation status
- Detect and **preserve** conflicts rather than resolving them silently
- Track confirmations as new revisions
- Retain provenance
- Serialize to JSONL or Parquet
- Produce model-ready snapshots

## Conflicts are kept, not resolved

```text
Patient reports cycle length:      52 days
Clinical note reports cycle length: 48 days
```

Both events persist. `conflict_resolution.py` emits an `EvidenceConflict`:

```python
EvidenceConflict(
    variable_name="Cycle length",
    canonical_variable_code="cycle_length",
    event_ids=[...],
    conflict_type="value_disagreement",
    recommended_resolution="Prefer the patient-confirmed value; ask at next contact.",
    requires_human_review=True,
)
```

A snapshot must still pick one value to hand a model. It does so under explicit,
documented rules — higher-trust provenance first, then more recent
`observed_at` — and it records `n_candidates` so a downstream reader can see the
choice was contested. The conflict travels *with* the snapshot; it is not
discarded once a winner is chosen.

Detected conflict types: `value_disagreement`, `unit_disagreement`,
`temporal_disagreement`, `presence_vs_negation`, `duplicate_measurement`.

## Snapshots

A snapshot is a parameterized, reproducible view — not the source of truth.

```python
snapshot = event_store.build_snapshot(
    patient_id="P001",
    as_of="2026-07-01",
    allowed_confirmation_statuses=["confirmed", "not_required"],
    include_modalities=["laboratory", "questionnaire", "ultrasound_image"],
)
```

It contains selected values, source event ids, recency in days, a missingness
mask, quality, conflicts, excluded events with reasons, and its creation
timestamp.

Two snapshots built with different confirmation policies are both valid, and
both reproducible from the same log. That is the point: "what would the model
see if we only trusted clinician-confirmed evidence?" is a query, not a
re-ingestion.

## The gate that matters

```python
@property
def is_model_ready(self) -> bool:
    return (
        self.missingness_status == "observed"
        and self.confirmation_status in {"confirmed", "not_required"}
    )
```

Unconfirmed speech, document, and imaging extractions are stored, queryable, and
reviewable — but they cannot reach a model until a human confirms them.
`tests/unit/test_speech_confirmation.py` asserts this directly.

## Condition-agnostic by construction

The store knows about *canonical variables*, not about PCOS. Supporting another
hormonal condition means adding registry entries and an adapter under
`models/adapters/` — the substrate does not change.
