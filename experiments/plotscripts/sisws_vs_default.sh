#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset21 $LOGBASE/preset22 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix sisws \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Local weight sharing on Catch" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "Default" "LWS T" "LWS S" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
