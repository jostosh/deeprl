#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset138 " " \
    --mode sweep \
    --trace_by lpq_gamma \
    --image_suffix policy_quantization_best \
    --no_log_scale \
    --xrange 1 256 \
    --yrange 0 1 \
    --title "Temperature Sweep GLPQ" \
    --xlabel "$ \gamma$" \
    --ylabel "Mean score" \
    --labels "A3C GLPQ" \
    --legend_at "upper left" \
    $*
