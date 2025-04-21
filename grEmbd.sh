#!/bin/bash
#SBATCH --time=1:10:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_1g.5gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm_output/train-%j.out  #%j for jobID

source /home/mohsenh/projects/def-ilie/mohsenh//ENV/prostT5/bin/activate
python amino_clust.py
