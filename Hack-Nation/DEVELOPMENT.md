# Development guide

## Environment

```bash
make install     # uv sync --all-extras, or venv + pip install -e ".[dev]"
```

Optional extras (`torch`, `xgboost`, `imaging`, `speech`, `documents`) are not
required. Every module has a documented fallback and CI runs the whole suite
without them in the `minimal-deps` job.

## Common commands

```bash
make format            # ruff format
make lint              # ruff check
make typecheck         # mypy
make test              # unit + contract
make integration-test  # cross-module, synthetic
make smoke             # tiny end-to-end model runs
make validate-registry # registry cross-reference checks
make docs              # mkdocs build --strict
make ci                # everything above
```

## Test layout

| Directory | Purpose | Must run without |
|:--|:--|:--|
| `tests/unit/` | One module, one behavior | Any real data, any optional dep |
| `tests/contract/` | Schemas and registries agree | Any real data |
| `tests/integration/` | Several modules together | Any real data |
| `tests/smoke/` | Tiny models train end to end | Any real data |
| `tests/fixtures/` | Synthetic generators and committed corpora | — |

Smoke-test metrics are never scientific results. They prove a forward pass runs.

## The tests that protect the invariants

Do not weaken these — they exist because the failure they catch is silent:

- `test_preprocessing_leakage.py` — preprocessing fitted on the training fold only
- `test_temporal_splits.py` — participant days never straddle train/test
- `test_speech_confirmation.py` — unconfirmed events are not model-ready
- `test_document_grounding.py` — ungrounded values are dropped, not added
- `test_prototype_language.py` — banned-phrase guard fires
- `test_ultrasound_quality_gate.py` — poor quality abstains from measurement
- `test_unit_conversions.py` — every factor in the registry

## Running an experiment

```bash
python scripts/train_static_baselines.py --config configs/experiments/exp_static_baselines.yaml
```

Every script is config-driven and writes a complete artifact directory:
resolved config, environment, git commit, data/split/feature manifests, training
log, metrics, predictions, checkpoint, figures, model card, README.

To reproduce someone's result you need only their commit and their config.

## Debugging tips

- **A registry error mentions two files.** `scripts/validate_registry.py` checks
  cross-references; the fix is usually in `variables.yaml`, not in the file that
  reported it.
- **A unit conversion raises rather than converting.** That is intended. Add the
  source unit to `registry/units.yaml` with a test; never bypass the converter.
- **An event refuses to validate.** Read the message — the schema states which
  invariant it protects. The fix is almost never to relax the schema.
- **A torch import fails.** It should not be imported at module scope anywhere.
  Move it inside the function and add a fallback.
