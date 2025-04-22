#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_output/train-%j.out  #%j for jobID

source /home/mohsenh/projects/def-ilie/mohsenh/ENV/prostt5ENV/bin/activate
python evaluation/dimension_reduction.py
