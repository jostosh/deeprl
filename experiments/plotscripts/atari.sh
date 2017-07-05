#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir ~/tensorflowlogs/cartesius/v0.9.7/Pong-v0/ \
    --mode mean \
    --image_suffix ff_vs_ww \
    --xrange 0 60 \
    --xlabel "Epoch" \
    --ylabel "Score" \
    --legend_at "upper left" \
    --subset_params model policy_quantization env glvq \
    --trace_by model policy_quantization \
    --fontsize 20 \
    --exclude policy_quantization a3c_lstm \
    $*

