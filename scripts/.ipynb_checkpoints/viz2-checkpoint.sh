#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --job-name="bash"

source ../../mario/MergeLM/unc2/bin/activate
python3 ../visualization.py --model_option MATH