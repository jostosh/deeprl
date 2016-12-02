#!/usr/bin/env bash

SLURM_JOB_CPUS_PER_NODE=5 CUDA_VISIBLE_DEVICES='' python3 deeprl/rlmethods/a3c_distributed.py --job_name ps \
--task_index 0 --env CartPole-v0 --log_dir /home/jos/tensorflowlogs/distributed/$1 --port0 $2 &

START=0
END=4
for (( c=$START; c<$END; c++ ))
do
    echo Starting worker $c
    SLURM_JOB_CPUS_PER_NODE=5 CUDA_VISIBLE_DEVICES='' python3 deeprl/rlmethods/a3c_distributed.py --env Breakout-v0 \
    --job_name worker --task_index $c --model a3c_ff --log_dir /home/jos/tensorflowlogs/distributed/$1 --port0 $2 \
    & \

    sleep 4
done

wait ${!}