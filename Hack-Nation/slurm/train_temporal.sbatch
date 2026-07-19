#!/usr/bin/env bash
#SBATCH --job-name=prism-temporal
#SBATCH --output=logs/temporal_%A_%a.out
#SBATCH --error=logs/temporal_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Step 9 — dynamic hormonal state. Array over seeds.
#
# Each array task runs ONE seed via --seed, writing its own directory. The script
# has no --fold flag: folds are enumerated from the config's `seeds:` list, so one
# seed per task is how you parallelise it.
#
# Splits are grouped by PARTICIPANT. Never introduce a day-level split here:
# adjacent days are near-duplicates and the resulting metric is meaningless.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/prism.env"
prism_activate_env
cd "$PRISM_REPO"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
OUTDIR="$PRISM_ARTIFACT_ROOT/temporal_${SLURM_ARRAY_JOB_ID:-local}/seed_${SEED}"
prism_record_provenance "$OUTDIR"

# mcPHASES is credentialed. Fail loudly rather than silently training on nothing.
if [[ ! -d "$PRISM_DATA_ROOT/mcphases" ]]; then
  echo "ERROR: $PRISM_DATA_ROOT/mcphases not found. mcPHASES requires credentialed access." >&2
  exit 1
fi

python scripts/prepare_mcphases.py \
  --config configs/data/mcphases.yaml \
  --data-root "$PRISM_DATA_ROOT/mcphases" \
  --output-dir "$OUTDIR"

# The missing-modality ablation runs by default; pass --skip-ablation to omit it.
python scripts/train_temporal.py \
  --config configs/experiments/exp_dynamic_state.yaml \
  --data-root "$PRISM_DATA_ROOT/mcphases" \
  --output-dir "$OUTDIR" \
  --seed "$SEED"
