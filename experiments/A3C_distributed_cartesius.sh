#!/usr/bin/env bash

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=17
#SBATCH --job-name=A3C_Distributed
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-%j.log
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=gpu_short

module load python/3.5.0
module load cuda/7.5.18
module load cudnn/7.5-v5
module load gcc/4.9.2

LOGS=/scratch-shared/jvdw/tensorflowlogs/distributed/$1/$2
srun -n 1 -N 1 --exclusive python3 mproj/deeprl/rlmethods/a3c_distributed.py --job_name ps --task_index 0 --env $1 \
--log_dir $LOGS --port0 $3 &

START=0
END=16
for (( c=$START; c<$END; c++ ))
do
    echo Starting worker $c
    srun -n 1 -N 1 --exclusive python3 mproj/deeprl/rlmethods/a3c_distributed.py --env $1 --job_name worker \
    --task_index $c --model a3c_lstm --log_dir $LOGS --port0 $3 &
    sleep 1
done

wait ${!}