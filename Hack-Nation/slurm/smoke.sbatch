#!/usr/bin/env bash
#SBATCH --job-name=prism-smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Run this FIRST on a new cluster. It uses only synthetic fixtures, touches no
# real data, and proves the environment works before you burn an allocation on
# a real job that fails at import time.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/prism.env"
prism_activate_env
cd "$PRISM_REPO"

python --version
python -c "import numpy, pandas, sklearn, pydantic; print('core deps OK')"
python -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')" \
  || echo "torch not installed — CPU fallbacks will be used"

make validate
python -m pytest tests/unit tests/contract -q
python -m pytest tests/smoke -q -m smoke

echo "Environment verified. Real jobs can be submitted."
