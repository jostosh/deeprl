#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset42 $LOGBASE/preset55 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix bias_init \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Bias intialization" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "Torch" "$\xi = 0.01$" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
