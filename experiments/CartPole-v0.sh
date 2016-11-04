#!/usr/bin/env bash
python3 a3c.py --env CartPole-v0 --model small_fcn --t_max 1024 --learning_rate 0.05 --input_shape 4 --n_threads 1 --lr_decay 0.25 --rms_epsilon 10e-8 --beta 0.0 --gamma 0.9
