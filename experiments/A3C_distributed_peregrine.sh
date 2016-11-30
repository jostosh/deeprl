#!/usr/bin/env bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --job-name=A3C_CartPole_distributed
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output job-%j.log
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=short

module load Python/3.5.1-foss-2016a
LOGS=/home/s2098407/tensorflowlogs/distributed/cartpole/run0002
srun -n 1 -N 1 python3 mproj/deeprl/rlmethods/a3c_distributed.py --job_name ps --task_index 0 --env CartPole-v0 --log_dir $LOGS &

START=0
END=4
for (( c=$START; c<$END; c++ ))
do
    echo Starting worker $c
    srun -n 1 -N 1 --exclusive python3 mproj/deeprl/rlmethods/a3c_distributed.py --env CartPole-v0 --job_name worker --task_index $c --model small_fcn --t_max 1024 --learning_rate 0.05 --input_shape 4 --n_threads $END --rms_decay 0.9 --rms_epsilon 1e-2 --beta 0.0001 --gamma 0.9 --log_dir $LOGS &
    sleep 1
done

wait ${!}