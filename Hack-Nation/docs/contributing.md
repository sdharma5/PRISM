# Contributing

The full guide lives in [CONTRIBUTING.md](https://github.com/AngelaNing1/Hack-Nation/blob/main/CONTRIBUTING.md)
at the repository root. The essentials:

## Before a PR

```bash
make ci   # format, lint, typecheck, test, validate-registry, docs
```

## Non-negotiables

1. No fabricated patient-level links across datasets.
2. Preprocessing fitted inside the training fold.
3. Longitudinal splits grouped by participant.
4. Missing never silently becomes zero.
5. Unconfirmed extracted events never reach a model.
6. Imaging output stays model-generated until clinician confirmation.
7. Research-oriented, probabilistic language — no diagnostic claims.
8. Abstention is always a valid output.
9. Schemas are versioned, never silently changed.
10. Docs update in the same PR.

## The question that blocks PRs

> Does this change alter any medical or scientific claim?

Answer it explicitly. If yes, state which claim and what evidence now supports
it.
