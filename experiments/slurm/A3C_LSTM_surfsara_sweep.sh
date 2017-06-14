#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=SWEEP
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-SWEEP-%j.log
#SBATCH --mem=2000

module load python/3.5.0
module load cuda/7.5.18
module load cudnn/7.5-v5
module load gcc/4.9.2

export PYTHONPATH="$PYTHONPATH:/home/jvdw/mproj"

srun python mproj/deeprl/experiments/param_sweep.py $*