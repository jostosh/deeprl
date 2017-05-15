#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=A3C_SISWS
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-SISWS-%j.log
#SBATCH --mem=2000

module load python/3.5.0
module load cuda/7.5.18
module load cudnn/7.5-v5
module load gcc/4.9.2

export PYTHONPATH="$PYTHONPATH:/home/jvdw/mproj"

srun python3 $HOME/mproj/deeprl/rlmethods/a3c.py --model a3c_sisws --n_threads 12 $*