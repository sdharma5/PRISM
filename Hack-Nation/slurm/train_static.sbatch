#!/usr/bin/env bash
#SBATCH --job-name=prism-static
#SBATCH --output=logs/static_%j.out
#SBATCH --error=logs/static_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Step 3 — static baselines. CPU only; this finishes faster than most queue
# waits, so run it locally unless you are batching it with everything else.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/prism.env"
prism_activate_env
cd "$PRISM_REPO"

OUTDIR="$PRISM_ARTIFACT_ROOT/static_${SLURM_JOB_ID:-local}"
prism_record_provenance "$OUTDIR"

python scripts/train_static_baselines.py \
  --config configs/experiments/exp_static_baselines.yaml \
  --data-root "$PRISM_DATA_ROOT" \
  --output-dir "$OUTDIR"

python scripts/build_model_card.py --experiment-dir "$OUTDIR" || true
echo "Artifacts: $OUTDIR"
