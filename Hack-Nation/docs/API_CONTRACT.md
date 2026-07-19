# PRISM API contract

The HTTP surface between the Python inference stack and the patient frontend.

**This is a research prototype. No endpoint here returns a diagnosis, and no
field in any response should be rendered as one.**

## Running it

```bash
# from the repository root
pip install -e ".[api,dev]"
uvicorn apps.api.main:app --reload --port 8000
```

Interactive schema: <http://localhost:8000/docs>.

| Environment variable | Effect |
|:--|:--|
| `PRISM_CORS_ORIGINS` | Comma-separated allowed origins. Defaults to the Next.js dev server on ports 3000. |
| `PRISM_EVENT_LOG` | Path to mirror the append-only event ledger to JSONL. Unset means process-lifetime memory only. |
| `PRISM_LOG_LEVEL` | Standard logging level. Defaults to `INFO`. |

## How models load

`apps/api/registry.py` builds a `ModelRegistry` **once**, in the FastAPI
lifespan handler, from `configs/models/inference_encoders.yaml`. Encoders are
never reloaded per request.

Startup **fails** if the static clinical branch cannot be loaded. That branch is
the only component entitled to issue a whole-patient PCOS score, so a service
without it cannot do the job it exists for. Every other branch is optional and
reports its own absence.

The registry never substitutes an untrained encoder for a trained one — it
delegates to `inference.encoder_registry.build_encoders`, which raises rather
than swapping in the heuristic segmenter, and adds the reporting side of that
contract through `/api/v1/models/status`.

The adapter is wired with `PrototypeSimilarityModel` and
`PhenotypeStabilityEngine`. Both are rule-based with working defaults, and
omitting them is not neutral: the adapter then returns empty
`profile_similarities` and no stability verdict, which a frontend reads as "no
similarity found" rather than "similarity was never computed".

## Endpoints

| Method | Path | Purpose |
|:--|:--|:--|
| `GET` | `/api/v1/health` | Liveness. Does not touch the models. |
| `GET` | `/api/v1/models/status` | Per-branch availability. **The frontend must key off this rather than hardcoding availability.** |
| `POST` | `/api/v1/patients/infer` | Main route. Runs whichever branches have input. |
| `POST` | `/api/v1/patients/infer/static` | Static branch only. |
| `POST` | `/api/v1/patients/infer/temporal` | Longitudinal branch only. Never yields a PCOS score. |
| `POST` | `/api/v1/patients/infer/ultrasound` | Gated off; returns `503` with a reason. |
| `POST` | `/api/v1/events` | Append events to the ledger. |
| `GET` | `/api/v1/events/{patient_id}` | Read a patient's events. Unknown patient returns `[]`, not `404`. |
| `POST`/`GET` | `/api/v1/jobs/{documents,speech,ultrasound}` | Ingestion job lifecycle. |

## Request rules

**Missing means absent, not zero.** Omit a field you have no value for. Sending
`0` for an unmeasured fasting glucose is a different clinical claim, and nothing
downstream can recover the distinction once the number is inside a token.

A `clinical_features` map containing only `null` values is rejected with `422`.
It is almost always a client that shipped an unfilled form template, and
failing loudly beats returning a confident-looking abstention.

Canonical codes must exist in `registry/variables.yaml`. Unknown codes are
dropped at the encoder boundary rather than carried as evidence.

Every nested record must name the same patient as the request. Mismatches are
`422` — including `temporal_observations`, whose identity field is
`participant_id` rather than `patient_id`.

## How a request becomes a `PatientDataBundle`

`PatientInferenceRequest.to_bundle()` in `apps/api/schemas/requests.py`:

1. `confirmed_events` pass through unchanged, keeping their own provenance.
2. `clinical_features` become events with provenance `patient_confirmed` and
   modality `questionnaire`. Null values are skipped, not encoded as events —
   an event asserts that an observation happened.
3. `temporal_observations` (internally `ParticipantDay`) become a `TemporalInput`.
4. The bundle validator rejects cross-patient contamination.

## Versioning

Every response carries `schema_version` (currently `1.0.0`). Additive optional
fields do not bump it; renames, removals and semantic changes to an existing
field do. A client should refuse a major version it does not recognise rather
than render fields it may be misreading.

`docs/openapi.yaml` is generated from the live app by
`python scripts/export_openapi.py` and is a **build input** — the frontend's
TypeScript types are generated from it rather than hand-written. Regenerate and
commit it whenever `apps/api/schemas/` changes; a stale spec silently produces
client types that disagree with the server.

## Design rules for this schema

Treat `apps/api/schemas/responses.py` as a public API:

- **Every field must be fillable.** A field the mapper cannot populate is worse
  than an absent one — a client binds to it and renders `null` forever.
  `test_every_declared_field_is_fillable` enforces this.
- **No bare dicts.** `dict[str, Any]` generates `unknown` in TypeScript and
  pushes the modelling burden onto the client.
- **Enumerate what is enumerable.** `Literal` becomes a real union type in
  generated clients; plain `str` becomes an unchecked string.
- **Absent is not zero.** `None` means not measured, `0.0` means measured and
  average. Never collapse them.

Note that `AxisStatus` deliberately omits `awaiting_confirmation`: nothing
currently emits it, and an enum member the backend cannot produce becomes dead
branches in every client. Add it in the same change that makes something emit it.

## Two different coverages

These are easy to confuse and mean different things:

| Field | Meaning |
|:--|:--|
| `modality_coverage` | Fraction of branches (static/temporal/ultrasound) that contributed. The "data coverage" of the profile header. |
| `pcos_assessment.feature_coverage` | Fraction of the *static model's own* inputs that were observed rather than imputed from training medians. |
| `phenotype.domain_scores[].coverage` | Weight-weighted fraction of one domain's variables that were observed. |

A score resting on 16% observed features is far weaker than the same number
resting on 90%, and only `feature_coverage` shows that.

## How internal output becomes a response

`inference/presentation/website_mapper.py` maps `PatientEvidenceReport` onto
`WebsitePCOSProfileResponse`. The internal `PCOSProfileOutput` is **never**
returned directly: it changes shape as models change, and it carries fields that
are misleading without their thresholds.

Three rules hold throughout the mapper:

1. **Never invent availability.** A branch that did not run says so and carries
   a reason.
2. **Qualifiers travel with the values they qualify.** Symptoms-only androgenic
   evidence carries that fact in the same object as the score, so no frontend
   can render one without the other.
3. **Absent is not zero.** `None` survives. A domain never assessed must not
   arrive as `0.0`, which reads as "exactly average".

### Domain scores

Each domain carries its registry `label`, `coverage`, `evidence_source`,
`observed_variables`, `missing_variables` and a `display_order`. Clients must
sort by `display_order` — dict iteration order is not a contract — and must not
keep their own key→label map, which would drift from the registry.

`scale` is always `cohort_z_score`. These are z-scores: `1.9` is 1.9 standard
deviations above the training cohort mean, **not** 190%.

### Supporting evidence

Internally an axis records threshold expressions like `"cycle_length > 35.0"`.
That is a machine expression, not a sentence, and must never be shown to a
patient as-is. The response splits it into `statement`, `variable_code`, `axis`
and `guideline_source` so the client renders prose. Statements concerning no
single variable (`"static ovarian composite z=+1.90"`) carry `variable_code:
null`.

### Evidence bands

`raw`/`calibrated` scores are banded, not percentaged:

| Score | Band |
|:--|:--|
| `< 0.25` | `low` |
| `< 0.50` | `moderate` |
| `< 0.75` | `elevated` |
| `>= 0.75` | `high` |
| `None` | `not_available` |

A score of `0.696` displays as **“PCOS-related evidence: Elevated / Model score:
0.70”**, never as “69.6% chance of PCOS”. `not_available` is not `low` — "low
evidence" is a finding, "not available" is the absence of one.

The band is computed from the calibrated score when present, so the band and the
displayed number cannot disagree.

### Temporal method translation

| Code | Display |
|:--|:--|
| `locf` | Based on the latest observed value |
| `ridge_window` | Estimated from the recent measurement pattern |
| `logistic` | Classified from recent longitudinal measurements |

`hormone_estimates` is keyed by **canonical variable code** (`urinary_lh`,
`e3g`, `pdg`), not a display abbreviation, so an estimate joins cleanly to the
events and registry entry for the same variable. Each entry also carries
`display_name` and `unit`.

LOCF carries the last observation forward unchanged. It must never be described
as a prediction. Unrecognised codes pass through as-is rather than borrowing
another method's wording.

## Missing encoders and partial results

A failure in one optional encoder does not fail the report. The orchestrator
collects per-encoder failures into `warnings`; the response still carries every
branch that did run, and `missing_modalities` names the rest.

| Situation | Behaviour |
|:--|:--|
| Static branch absent | `pcos_assessment.available = false` with `unavailable_reason`. |
| Temporal absent | `current_state.available = false` with a reason. |
| Ultrasound requested | `503` naming the gate. |
| One encoder throws | Report returns; the failure appears in `warnings`. |

## Ultrasound

Disabled in `configs/models/inference_encoders.yaml`. The checkpoint **exists
and loads** — this is a deliberate gate, not an absence, which is why
`/models/status` reports `trained: true, persisted: true,
validated_for_inference: false` with a reason naming the uncorrected ovary
segmentation head and the oracle-assisted follicle Dice.

Uploads are accepted so the interface can be built; jobs terminate as
`unavailable` carrying that same reason. The oracle-assisted Dice is never a
patient-visible result.

To enable later: validate an end-to-end checkpoint on held-out data, set
`enabled: true`, and the branch reports itself available on the next start. No
frontend change is required — availability is data, not code.

## Invariants

Asserted in `tests/unit/test_api_inference.py` against real loaded encoders:

- Temporal input alone never produces a PCOS model score.
- Ultrasound alone never produces a PCOS model score.
- An indeterminate or unstable phenotype never names a dominant profile — while
  still returning the similarities behind the withheld label.
- A symptoms-only androgenic result always carries its qualifier and states that
  biochemical evidence was unavailable.
