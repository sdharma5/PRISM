# Model cards

Every model exports `ModelCardMetadata` via
`BasePrismModel.export_model_card_metadata()`, and every experiment writes a
`model_card.json` into its artifact directory.

```bash
python scripts/build_model_card.py --experiment-dir artifacts/experiments/<ID>
```

This generates or updates `artifacts/model_cards/<model_name>.md`. It never
overwrites the human-reviewed top-level `MODEL_CARD.md` — a generated file
cannot review its own claims.

## Required fields

| Field | Contains |
|:--|:--|
| `intended_use` | The research question the model addresses |
| `out_of_scope_uses` | Explicit, including every clinical use |
| `training_datasets` / `evaluation_datasets` | Registry ids and versions |
| `metrics` | Discrimination **and** calibration |
| `limitations` | Sample size, population, missingness, stability |
| `ethical_considerations` | Subgroup performance, symptom-vs-biochemical substitution |
| `non_diagnostic_statement` | Fixed, non-removable |

## The non-diagnostic statement

Carried by every model card and not overridable:

> This model is a research artifact. It does not diagnose any condition, does
> not provide medical advice, and is not validated for clinical use.
