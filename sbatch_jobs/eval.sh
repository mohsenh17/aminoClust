#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_output/eval-%j.out  #%j for jobID

source /home/mohsenh/projects/def-ilie/mohsenh/ENV/prostt5ENV/bin/activate
python evaluation/cluster_eval.py
