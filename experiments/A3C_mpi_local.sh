#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES='' mpiexec -n 8 python3 a3c_mpi.py --logdir /home/jos/tensorflowlogs/mpi/Breakout-v0/run0000
