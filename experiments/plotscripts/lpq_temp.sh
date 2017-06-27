#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset89 " " \
    --mode sweep \
    --trace_by lpq_temp \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -1 2 \
    --yrange 0 1 \
    --title "Temperature Sweep LPQ" \
    --xlabel "$\log_{10}(\gamma)$" \
    --ylabel "Mean score" \
    --labels "A3C LPQ" \
    --legend_at "upper left" \
    $*
