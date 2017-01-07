#!/usr/bin/env bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=A3C_LSTM_v09
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-A3C_LSTM_v09-%j.log
#SBATCH --mem=2000
#SBATCH --partition=gpu_short

module load python/3.5.0
module load cuda/7.5.18
module load cudnn/7.5-v5
module load gcc/4.9.2

srun python3 $HOME/mproj/deeprl/rlmethods/a3c.py --model a3c_lstm --n_threads 16 --clip_rewards $*