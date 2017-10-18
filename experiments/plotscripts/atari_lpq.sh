#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.8/Pong-v0/ \
    --mode mean \
    --image_suffix atari_lpq \
    --xrange 0 60 \
    --xlabel "Epoch" \
    --ylabel "Score" \
    --legend_at "upper left" \
    --subset_params model policy_quantization env glvq zpi \
    --trace_by model policy_quantization zpi \
    --fontsize 20 \
    --exclude a3c_ff_ss a3c_ff_ww a3c_sisws a3c_lstm zpi \
    $*
