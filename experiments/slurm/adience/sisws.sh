#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=ADIENCE_SISWS_TEST
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-ADIENCE_SISWS_TEST-%j.log
#SBATCH --mem=8000
#SBATCH --partition=short

module load tensorflow/1.0.1-foss-2016a-Python-3.5.2

srun python mproj/deeprl/experiments/adience/train.py \
    --model spatial \
    --n_epochs 200 \
    --random_inits 3 \
    --datadir
