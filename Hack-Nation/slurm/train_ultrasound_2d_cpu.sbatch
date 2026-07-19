#!/usr/bin/env bash
#SBATCH --job-name=prism-us2d-cpu
#SBATCH --output=logs/us2d_cpu_%j.out
#SBATCH --error=logs/us2d_cpu_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# 2D slice model on CPU with real parallelism.
#
# The login node has ONE core, so the interactive 2D run is effectively serial.
# This job requests 16 and points torch at all of them. A 672K-parameter 2D
# U-Net on 128x128 patches is comfortably CPU-trainable at that width; the GPU
# queue depth (100+ pending per partition) makes this the faster path to a real
# 2D checkpoint.
#
# IMPORTANT: writes to a SEPARATE output directory. The interactive run is still
# going and both would otherwise write checkpoint_best.pt to the same path, with
# the later writer silently winning and the reported metrics belonging to
# whichever process happened to finish second.

set -euo pipefail

REPO="/weka/scratch/uchitra1/users/a_s/mit_hackathon/Hack-Nation"
cd "$REPO"
mkdir -p logs

# torch reads these at import; setting them after import has no effect.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"
"$REPO/.venv/bin/python" -c "import torch; torch.set_num_threads(int('${SLURM_CPUS_PER_TASK:-16}')); print('torch', torch.__version__, 'threads', torch.get_num_threads())"

"$REPO/.venv/bin/python" scripts/train_ultrasound_encoder.py \
  --config configs/experiments/exp_usova3d_2d_unet.yaml \
  --output-dir artifacts/experiments/exp_usova3d_2d_unet_slurm

"$REPO/.venv/bin/python" scripts/evaluate_ultrasound_encoder.py \
  --experiment-dir artifacts/experiments/exp_usova3d_2d_unet_slurm
