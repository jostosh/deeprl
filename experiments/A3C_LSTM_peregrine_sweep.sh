#!/usr/bin/env bash
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=A3C_FF_TEST_v09
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-A3C_FF_TEST_v09-%j.log
#SBATCH --mem=2000

module load Python/3.5.1-foss-2016a

srun python mproj/deeprl/experiments/param_sweep.py $*