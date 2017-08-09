#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=A3C_FF
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-FF-%j.log
#SBATCH --mem=2000

module load tensorflow
source $HOME/envs/mproj10/bin/activate


srun python3 $HOME/mproj/deeprl/rlmethods/a3c.py --model a3c_ff --n_threads 12 $*