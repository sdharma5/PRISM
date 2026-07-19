#!/bin/bash
#SBATCH --job-name=us_train
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p l40s,a100,nvl,h100
#SBATCH --time=2:00:00
#SBATCH --output=/weka/scratch/uchitra1/users/a_s/ultrasound_boxer/train_%j.log

cd /weka/scratch/uchitra1/users/a_s/ultrasound_boxer
python3 train.py data/MMOTU/OTU_2d --out us_classifier.pt --epochs 30
