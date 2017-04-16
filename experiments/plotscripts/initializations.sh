#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset42 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix sweep_multiple_test \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Parameter initializations" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF torch" "A3C FF default" \
    --legend_at "upper left" \
    $*
