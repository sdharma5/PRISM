# Speech pipeline (Step 6)

Converts spoken symptoms and history into **reviewable** structured events.

## Pipeline

```text
Audio → quality assessment → speech-to-text → speaker identification →
medical event extraction → negation and temporality detection →
evidence-span linking → human review → confirmed events → symptom token
```

Modes: patient narration and clinician dictation are prioritized;
patient–clinician conversation is experimental.

## Extraction is meaning-preserving

| Input | Output |
|:--|:--|
| "My periods have been between 45 and 70 days apart for about a year." | `cycle_irregularity`, present, duration ≈ 12 months, current |
| "I had acne in high school, but I do not have it now." | `acne`, **historical_resolved**, current = false |
| "My sister has PMOS." | `family_history_pmos`, present, current = false |

The third case is the one that matters most: a relative's diagnosis must never
become the patient's. It is a canonical variable of its own, and a unit test
asserts the distinction holds.

Negation, historicality and uncertainty are detected with cue-plus-scope
(ConText-style) logic, and every event carries `evidence_text` plus start/end
seconds.

## Adapters, not hard dependencies

Transcription and extraction are pluggable. The defaults are offline: a scripted
transcription adapter and a rule-based extractor driven by a documented lexicon.
Whisper and LLM adapters exist behind the same interface and are never in the
test path — CI runs with no network and no API key.

## Confirmation

```json
{ "confirmed": [], "awaiting_confirmation": [], "rejected": [] }
```

Only confirmed events enter the model-ready event store. Extraction quality is
measured on the *pre*-confirmation output; modeling uses only *post*-confirmation
events.

## Metrics

Word error rate · symptom extraction precision/recall/F1 · negation F1 ·
temporality F1 · medication-event F1 · speaker-attribution accuracy ·
unsupported-event rate · user-correction rate.

Evaluated on a committed synthetic scripted corpus covering present, negated,
historical and uncertain symptoms, family history, medication changes, cycle
timing, fertility goals, clinician questions, speaker confusion and approximate
dates. It is fictional and supports no real-world performance claim.
