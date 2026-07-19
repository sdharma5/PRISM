#!/usr/bin/env bash
#SBATCH --job-name=prism-us-enc
#SBATCH --output=logs/us_encoder_%j.out
#SBATCH --error=logs/us_encoder_%j.err
#SBATCH --time=04:00:00
# Multiple partitions: SLURM starts the job on whichever frees up first. The
# model is ~1.9M parameters on 16 volumes, so any of these GPUs is far more than
# sufficient -- queue wait dominates runtime, and pinning to one partition just
# means waiting for that partition.
#SBATCH --partition=l40s,a100,h100,nvl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Train the dual-head ovary/follicle segmentation model on USOVA3D.
#
# Self-contained on purpose: slurm/prism.env still carries placeholder paths
# (PRISM_REPO defaults to $HOME/prism-hormonal-health, which does not exist
# here), so sourcing it would send the job to the wrong tree.
#
# The whole cohort is 16 volumes and the model is ~1.9M parameters, so this is a
# short job. A 4-hour limit is generous; it exists to catch a hang, not because
# the run needs it.

set -euo pipefail

REPO="/weka/scratch/uchitra1/users/a_s/mit_hackathon/Hack-Nation"
cd "$REPO"
mkdir -p logs

echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$REPO/.venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

"$REPO/.venv/bin/python" scripts/train_ultrasound_encoder.py \
  --config configs/experiments/exp_usova3d_3d_unet.yaml

"$REPO/.venv/bin/python" scripts/evaluate_ultrasound_encoder.py \
  --experiment-dir artifacts/experiments/exp_usova3d_3d_unet
