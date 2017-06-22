#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset21 $LOGBASE/preset22 $LOGBASE/preset102 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix sisws \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "SISWS models vs. default A3C" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "Default" "SISWS T" "SISWS S" "SISWS New" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
