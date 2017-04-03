#!/usr/bin/env bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=A3C_FF_DEFAULT
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-A3C_FF_DEFAULT-%j.log
#SBATCH --mem=2000

module load Python/3.5.1-foss-2016a

srun python mproj/deeprl/rlmethods/a3c.py \
    --model a3c_ff \
    --n_threads 12 \
    --clip_rewards $*