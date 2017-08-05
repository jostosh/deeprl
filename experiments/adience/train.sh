#!/usr/bin/env bash
python3 train.py \
    --model=spatial \
    --trainable_centroids \
    --per_feature \
    --n_epochs 100 \
    --random_inits 3 \
    --datadir /home/jos/datasets/aligned \
    --first_normal \
    --logbase ~/tensorflowlogs/v0.9.7/adience
