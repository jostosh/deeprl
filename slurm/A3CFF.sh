#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=A3C_FF
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-FF-%j.log
#SBATCH --mem=2000
#SBATCH --partition=short

module load tensorflow

source $HOME/envs/mproj10/bin/activate

srun python3 $HOME/mproj/deeprl/train.py \
    --model a3cff \
    --n_threads 16 \
    --entropy_beta 0.01 \
    --T_max 1000000 \
    $*