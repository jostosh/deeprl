#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir ~/tensorflowlogs/cartesius/v0.9.7/Breakout-v0 \
    --mode mean \
    --image_suffix atari_june \
    --xrange 0 80 \
    --xlabel "Epoch" \
    --ylabel "Score" \
    --legend_at "upper left" \
    --subset_params model policy_quantization env \
    --trace_by model policy_quantization \
    $*
