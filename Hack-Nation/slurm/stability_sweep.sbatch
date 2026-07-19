#!/usr/bin/env bash
#SBATCH --job-name=prism-stability
#SBATCH --output=logs/stability_%A_%a.out
#SBATCH --error=logs/stability_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-9

# Step 5 — clustering + stability. The heaviest CPU step and embarrassingly
# parallel over seeds, so this is an ARRAY job: one seed per task.
#
# Each task writes its own directory. Aggregate afterwards with
# scripts/run_stability_analysis.py --aggregate.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/prism.env"
prism_activate_env
cd "$PRISM_REPO"

SEED="${SLURM_ARRAY_TASK_ID:-0}"
OUTDIR="$PRISM_ARTIFACT_ROOT/stability_${SLURM_ARRAY_JOB_ID:-local}/seed_${SEED}"
prism_record_provenance "$OUTDIR"

echo "Seed ${SEED} of array ${SLURM_ARRAY_TASK_COUNT:-1}"

python scripts/discover_phenotypes.py \
  --config configs/experiments/exp_subtype_stability.yaml \
  --data-root "$PRISM_DATA_ROOT" \
  --output-dir "$OUTDIR" \
  --seed "$SEED"

python scripts/run_stability_analysis.py \
  --config configs/experiments/exp_subtype_stability.yaml \
  --data-root "$PRISM_DATA_ROOT" \
  --output-dir "$OUTDIR" \
  --seed "$SEED"
