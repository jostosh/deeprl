#!/usr/bin/env bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=17
#SBATCH --job-name=A3C_Distributed
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-%j.log
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=short

module load Python/3.5.1-foss-2016a
LOGS=/data/s2098407/tensorflowlogs/mpi/$1/$2

srun python3 mproj/deeprl/rlmethods/a3c_mpi.py --env $1 --logdir $LOGS
