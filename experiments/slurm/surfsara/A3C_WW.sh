#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=A3C_SS
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-WW-%j.log
#SBATCH --mem=4000

module load python/3.5.0
module load cuda/7.5.18
module load cudnn/7.5-v5
module load gcc/4.9.2

export PYTHONPATH="$PYTHONPATH:/home/jvdw/mproj"

srun python3 $HOME/mproj/deeprl/rlmethods/a3c.py --model a3c_ff_ww --ss_temp 1.0  --n_threads 12 --ss_temp_global $*