#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=ADIENCE_DEFAULT
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-ADIENCE_DEFAULT-%j.log
#SBATCH --mem=8000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow/1.0.1-foss-2016a-Python-3.5.2
source envs/adience/bin/activate
srun python mproj/deeprl/experiments/adience/train.py \
    --model default \
    --n_epochs 200
