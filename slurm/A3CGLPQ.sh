#!/usr/bin/env bash
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=A3C_GLPQ
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-GLPQ-%j.log
#SBATCH --mem=2000

module load Python/3.5.1-foss-2016a

srun python3 $HOME/mproj/deeprl/train.py \
    --model a3c_ff \
    --n_threads 12 \
    --policy_quantization \
    --pq_cpa \
    --glvq \
    --beta 0.01 \
    --pi_loss_correct \
    --zpi \
    $*