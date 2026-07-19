# Contributing

## Before you start

Read [ARCHITECTURE.md](ARCHITECTURE.md) and the four ADRs in
[`docs/decisions/`](docs/decisions/). They encode constraints that look like
style preferences but are scientific requirements — most rejected PRs violate
one of them without realizing it.

## Setup

```bash
make install
make ci        # format, lint, typecheck, test, validate-registry, docs
```

`make ci` must pass before a task is considered complete.

## The twenty rules

1. Implement only Steps 1–9 from [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
2. Do not implement clinician search, phone scripts, PDF generation, treatment
   recommendations, or care navigation.
3. Never fabricate patient-level links across unrelated datasets.
4. Use strict typed schemas for every module boundary.
5. Preserve provenance and missingness for every observation.
6. Create small synthetic fixtures for all tests.
7. Do not commit restricted or personally identifiable health data.
8. Every new model ships with a config, training script, evaluation script,
   tests, documentation, and model-card metadata.
9. Every preprocessing operation is fitted inside the training fold.
10. Every longitudinal evaluation split is grouped by participant.
11. Every speech or document event links to source evidence.
12. Unconfirmed extracted events never enter model-ready snapshots.
13. Ultrasound output distinguishes model-generated from clinician-confirmed.
14. PCOS subtype output uses probabilistic, research-oriented language.
15. Every model supports an indeterminate or abstain output.
16. Run `make ci` before considering a task complete.
17. Update the relevant docs in the same pull request.
18. Never change a schema silently — increment its version and document it.
19. Keep modules independently runnable.
20. Prefer simple, tested baselines before complex neural architectures.

## Commit convention

```text
feat:      new capability
fix:       bug fix
docs:      documentation only
data:      dataset registry, mappings, ingestion
model:     model architecture or training
eval:      metrics and evaluation
refactor:  no behavioral change
test:      tests only
chore:     tooling, CI, dependencies
```

Release notes are generated from these prefixes.

## Pull requests

Every PR answers:

- What changed?
- Why is it scientifically valid?
- Which module is affected?
- Which schema or data contract changed?
- Which tests were added?
- Which documentation was updated?
- **Does this change alter any medical or scientific claim?**

A model PR additionally includes a config file, tests, documentation, the
artifact contract, metric definitions, known limitations, and a reproduction
command.

## Adding a canonical variable

1. Add it to `registry/variables.yaml` with type, unit, domain, valid range.
2. Add conversions to `registry/units.yaml` if it has more than one unit — and a
   test in `tests/unit/test_unit_conversions.py`. Every factor needs a test.
3. Map source columns in the relevant adapter's `mapping.py`.
4. If it belongs to a phenotype domain, add it to
   `registry/phenotype_domains.yaml` with a documented weight and direction.
5. `make validate-registry`.

## Adding a dataset

Open a dataset-request issue first. The registry entry must state
`allowed_uses` and `prohibited_claims` before any adapter is written — deciding
what a dataset may claim *after* seeing results is how unsupported claims happen.

## Language rules

Outputs are research phenotype profiles, evidence summaries, morphology
measurements, model-estimated probabilities, missing-information analyses, and
stability analyses.

They are never confirmed diagnoses, medical advice, treatment recommendations,
required clinical tests, guaranteed subtypes, or validated clinical decision
support. `models/phenotype/prototype_mapping.py` enforces a banned-phrase list
in code; the test suite asserts it fires.
