#!/bin/bash
#SBATCH --job-name=prism-build
#SBATCH --output=/weka/scratch/uchitra1/users/a_s/mit_hackathon/UI/prism-build-%j.log
#SBATCH --error=/weka/scratch/uchitra1/users/a_s/mit_hackathon/UI/prism-build-%j.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --partition=cpu

set -euo pipefail

export PATH="/weka/scratch/uchitra1/users/a_s/node_env/bin:$PATH"
export npm_config_cache="/weka/scratch/uchitra1/users/a_s/.npm-cache"

cd /weka/scratch/uchitra1/users/a_s/mit_hackathon/UI/prism-app

echo "=== Node $(node --version)  npm $(npm --version) ==="
echo "=== npm install ==="
npm install --prefer-offline

echo "=== next build ==="
npm run build

echo "=== Done. Start with: ==="
echo "    PATH=/weka/scratch/uchitra1/users/a_s/node_env/bin:\$PATH npm start --prefix /weka/scratch/uchitra1/users/a_s/mit_hackathon/UI/prism-app"
