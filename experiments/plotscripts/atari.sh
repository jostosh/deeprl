#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.8/Pong/ \
    --mode mean \
    --image_suffix pong0.9.8 \
    --xrange 0 80 \
    --xlabel "Epoch" \
    --ylabel "Score" \
    --legend_at "upper left" \
    --subset_params model policy_quantization env glvq value_loss_fac \
    --trace_by model policy_quantization \
    --fontsize 20 \
    --exclude a3c_lstm a3c_ff_ss a3c_sisws \
    $*

