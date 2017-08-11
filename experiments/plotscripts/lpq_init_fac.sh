#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset141 $LOGBASE/preset142 \
    --mode sweep \
    --trace_by lpq_init_fac \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -4 4 \
    --yrange 0 1 \
    --title "Temperature Sweep LPQ" \
    --xlabel "$\log_{10}(\lambda_{\text{LPQ init}})$" \
    --ylabel "Mean score" \
    --labels "A3C random uniform" "A3C zero clip"  \
    --legend_at "upper left" \
    $*
