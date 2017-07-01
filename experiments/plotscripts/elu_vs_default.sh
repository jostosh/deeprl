#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset4 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix sweep_multiple_test \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Activation Functions" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "ReLU" "ELU" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
