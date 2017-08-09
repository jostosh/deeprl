#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=A3C_FF_TEST_v09
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-A3C_FF_TEST_v09-%j.log
#SBATCH --mem=2000

module load tensorflow
source $HOME/envs/mproj10/bin/activate


srun python mproj/deeprl/experiments/param_sweep.py $*