#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem=400G
#SBATCH --job-name="bash"
#SBATCH -p compsci-gpu

source ../mario/MergeLM/unc2/bin/activate
python3 get_percentages.py --model_option 1