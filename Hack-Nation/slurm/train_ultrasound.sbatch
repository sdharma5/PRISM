#!/usr/bin/env bash
#SBATCH --job-name=prism-us
#SBATCH --output=logs/ultrasound_%j.out
#SBATCH --error=logs/ultrasound_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Step 8 — ultrasound, 2D-primary.
#
# NOTE: the three-stage training strategy in configs/models/ultrasound_segmentation.yaml
# is a DECLARED PLAN, not an implemented one — train_ultrasound.py fits no weights
# and has no --stage flag. This template runs what actually exists: preparation,
# then evaluation over the acquisition pathways.
#
# Restrict to one pathway with: sbatch --export=ALL,MODES=cine_loop ...

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/prism.env"
prism_activate_env
cd "$PRISM_REPO"

OUTDIR="$PRISM_ARTIFACT_ROOT/ultrasound_${SLURM_JOB_ID:-local}"
prism_record_provenance "$OUTDIR"

nvidia-smi || echo "WARNING: no GPU visible — this will fall back to CPU and be very slow."

MODES="${MODES:-single_frame cine_loop volume_3d}"

python scripts/prepare_ultrasound.py \
  --config configs/data/ultrasound.yaml \
  --data-root "$PRISM_DATA_ROOT" \
  --output-dir "$OUTDIR"

# shellcheck disable=SC2086 - MODES is a deliberate word-split list.
python scripts/train_ultrasound.py \
  --config configs/experiments/exp_ultrasound.yaml \
  --data-root "$PRISM_DATA_ROOT" \
  --output-dir "$OUTDIR" \
  --modes $MODES

# The gate metric that must stay at zero: it counts images accepted for
# measurement that could not support one.
python - <<'PY'
import json, os, pathlib, sys
p = pathlib.Path(os.environ["PRISM_ARTIFACT_ROOT"]) / f"ultrasound_{os.environ.get('SLURM_JOB_ID','local')}" / "metrics.json"
if p.exists():
    m = json.loads(p.read_text()).get("aggregate_metrics", {})
    rate = m.get("quality_gate_unsafe_acceptance_rate")
    print(f"quality_gate_unsafe_acceptance_rate = {rate}")
    if rate is not None and rate > 0:
        print("ERROR: the quality gate accepted images it should have refused.", file=sys.stderr)
        sys.exit(1)
PY
