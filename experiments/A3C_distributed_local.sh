#!/usr/bin/env bash

SLURM_JOB_CPUS_PER_NODE=4 CUDA_VISIBLE_DEVICES='' python3 deeprl/rlmethods/a3c_distributed.py --job_name ps --task_index 0 --env CartPole-v0 --log_dir /home/jos/tensorflowlogs/distributed/$1 &

START=0
END=4
for (( c=$START; c<$END; c++ ))
do
    echo Starting worker $c
    SLURM_JOB_CPUS_PER_NODE=4 CUDA_VISIBLE_DEVICES='' python3 deeprl/rlmethods/a3c_distributed.py --env CartPole-v0 \
    --job_name worker --task_index $c --model small_fcn --t_max 1024 --learning_rate 0.01 --input_shape 4 --n_threads 4 \
    --rms_decay 0.9 --rms_epsilon 1e-2 --beta 0.0001 --gamma 0.9 --log_dir /home/jos/tensorflowlogs/distributed/$1 &
    sleep 4
done

wait