# ADR-001: An append-only event store as the single evidence substrate

**Status:** Accepted · **Date:** 2026-07-18

## Context

Five modalities produce information about a person at different times, with
different reliability, in different units, some of it contradictory. The
conventional design is a wide patient table: one row per person, one column per
variable, last write wins.

That design destroys exactly the information PRISM needs. It cannot express
"the patient said 52 days and the clinical note said 48"; it cannot say *why* a
cell is empty; it cannot distinguish a measured value from an LLM-extracted one;
and it silently overwrites the earlier of two disagreeing measurements.

## Decision

The substrate is an **append-only store of `HormonalHealthEvent` records**. One
event is one observation, carrying value, unit, time, modality, provenance,
extraction confidence, confirmation status, missingness status, and an evidence
span or source location.

Consequences of "append-only":

- Events are never mutated or deleted. A confirmation appends a new revision.
- Conflicting events both persist. `conflict_resolution.py` *detects and labels*
  conflicts; it does not silently pick a winner.
- Model-ready inputs come from `build_snapshot(...)`, which selects values under
  explicit rules (allowed confirmation statuses, included modalities, an `as_of`
  time) and records what it excluded and why.

The wide table still exists — but as a *derived, parameterized view*, not as the
source of truth. Two snapshots with different confirmation policies are both
legitimate, and both are reproducible from the same event log.

## Consequences

**Cost.** More storage, more code, and every consumer must go through the
snapshot API rather than reading a DataFrame directly.

**Benefit.** Provenance survives to the metric table. Any number in a result can
be traced back to the sentence in the transcript or the page in the PDF it came
from. Missingness analysis becomes a first-class query rather than a
reconstruction after the fact.

**Condition-agnostic by construction.** The store knows about *variables*, not
about PCOS. Adding another hormonal condition means adding registry entries and
an adapter under `models/adapters/`, not reshaping the substrate.
