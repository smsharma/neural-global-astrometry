#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=03:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/neural-global-astrometry/

# python -u simulate.py -n 100 --name train_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-astrometry/
# python -u simulate.py -n 100 --name test_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-astrometry/
# python -u simulate.py -n 10 --name test_150_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-astrometry/ --f_sub 150
python -u simulate.py -n 100 --name test_225_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-astrometry/ --f_sub 225
