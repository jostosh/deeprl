#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=A3C_LSTM_v09
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-A3C_LSTM_v09-%j.log
#SBATCH --mem=2000
#SBATCH --partition=short

module load Python/3.5.1-foss-2016a

srun python mproj/deeprl/rlmethods/a3c.py --model a3c_lstm --n_threads 12 --clip_rewards $*