# Implementation plan

Steps 1–9. Each milestone has a definition of done that is checkable, not
aspirational.

## Excluded from this phase

Clinician contact search · insurance filtering · phone scripts · doctor-facing
PDF · treatment recommendations · care navigation · a trained cross-dataset
fusion model · any claim of clinical diagnosis or deployment.

## Milestone 1 — Repository foundation

pyproject and environment · Makefile · schemas · dataset registry · validation
scripts · CI · documentation skeleton · synthetic fixtures.

**Done when:** `make ci` passes.

## Milestone 2 — Static PMOS baseline

PMOS tabular loader · canonical mappings · split manifests · logistic regression
· XGBoost · metrics · calibration plots · baseline model card.

**Done when:** reproducible held-out metrics and saved predictions exist.

## Milestone 3 — Phenotype domains

Domain registry · composite scores · masked tabular autoencoder · static token
export · masked-reconstruction evaluation · documented limitations.

**Done when:** every participant receives domain scores, coverage, and an
embedding.

## Milestone 4 — Subtype stability

Clustering benchmark · bootstrap stability · feature ablation · perturbation
testing · indeterminate logic · abstention logic · stability report export.

**Done when:** subtype output includes probabilities, stability, and warnings.

## Milestone 5 — Speech

Audio upload · transcription adapter · structured extraction schema ·
confirmation state · scripted evaluation set · extraction metrics · symptom
token.

**Done when:** confirmed speech events enter the event store with evidence
spans.

## Milestone 6 — Documents

PDF text/table parser · laboratory extraction · units registry · source-page
grounding · confirmation · document evaluation set · document token.

**Done when:** verified laboratory events retain original **and** normalized
values.

## Milestone 7 — Ultrasound

Loader · de-identification checks · image QC · ovary segmentation · follicle
segmentation · instance extraction · morphology measurement · overlays ·
ultrasound token.

**Done when:** the module reports segmentation and measurement metrics.

## Milestone 8 — Dynamic hormonal state

Participant-day tables · grouped splits · GRU · hormone head · cycle-phase head
· symptom head · missing-modality ablations · temporal token.

**Done when:** the current-state representation is evaluated on held-out
participants.

## Milestone 9 — Repository integration

Standardize all token outputs · integration tests · example synthetic patient ·
documentation pages · architecture diagram · model cards · release candidate.

**Done when:** all five tokens serialize under one shared contract.

## Standing rules

1. Implement only Steps 1–9.
2. Never fabricate patient-level links across unrelated datasets.
3. Strict typed schemas at every module boundary.
4. Provenance and missingness preserved for every observation.
5. Small synthetic fixtures for all tests.
6. No restricted or identifiable health data committed.
7. Every model ships config, training script, evaluation script, tests, docs,
   and model-card metadata.
8. Preprocessing fitted inside the training fold.
9. Longitudinal splits grouped by participant.
10. Speech and document events link to source evidence.
11. Unconfirmed events stay out of model-ready snapshots.
12. Imaging output distinguishes model-generated from clinician-confirmed.
13. Probabilistic, research-oriented subtype language.
14. Indeterminate and abstain are supported outputs.
15. `make ci` before a task is complete.
16. Docs updated in the same PR.
17. Schemas versioned, never silently changed.
18. Modules independently runnable.
19. Simple tested baselines before complex architectures.
