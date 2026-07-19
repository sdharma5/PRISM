# Security and data-handling policy

## Reporting a vulnerability

Report security issues privately via GitHub Security Advisories on this
repository, or to the maintainers listed in `.github/CODEOWNERS`. Please do not
open a public issue for a vulnerability. Expect an acknowledgement within five
working days.

If a report involves exposed health data, say so in the first line. Those are
triaged ahead of everything else.

## What must never enter this repository

This is the practical core of the policy:

- **No patient-level clinical data**, identifiable or de-identified.
- **No raw audio**, even synthetic-sounding. Speech fixtures are transcripts.
- **No DICOM, NIfTI, or other imaging volumes.**
- **No PDFs of real reports**, redacted or otherwise.
- **No credentials, API keys, or dataset access tokens.** Use `.env`, which is
  gitignored; `.env.example` documents the variable names only.

`.gitignore`, a pre-commit hook, and the `no-clinical-data` CI job each enforce
this independently. Three overlapping checks is deliberate — the failure mode
here is not recoverable by a later commit, because git history retains it.

If clinical data is committed by accident, treat it as a disclosure incident:
notify the maintainers immediately, do not simply push a deletion commit, and
expect history rewriting plus credential rotation to be required.

## Handling restricted datasets

Several registry datasets are access-restricted (mcPHASES via credentialed
PhysioNet access, among others). Contributors are responsible for:

- obtaining access under their own credentials and data-use agreement;
- storing data outside the repository tree (`PRISM_DATA_ROOT`);
- honouring redistribution restrictions — a derived artifact can still leak the
  source, so aggregate outputs are not automatically shareable;
- never committing derived files that retain patient-level granularity.

Split manifests contain patient identifiers and are therefore also
dataset-scoped: `artifacts/splits/` is tracked only for synthetic cohorts.

## Model outputs are not clinical outputs

PRISM produces research artifacts. No output of this repository is validated for
clinical decision-making, and framing an output as diagnosis, triage, or
treatment guidance is a policy violation as well as a scientific error — see
`docs/decisions/ADR-003-subtype-language.md` and
`docs/decisions/ADR-004-human-confirmation.md`.

## Dependency and supply-chain hygiene

- Dependencies are pinned via the lockfile; Dependabot proposes updates weekly.
- Heavy and network-capable dependencies (transcription backends, PDF parsers)
  are **optional extras**, so the default install and the entire test path run
  offline.
- No module in the test path performs a network call. The speech and document
  pipelines ship deterministic offline adapters precisely so CI never needs an
  API key.
