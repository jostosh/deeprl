#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset47 $LOGBASE/preset48 $LOGBASE/preset49 $LOGBASE/preset50 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization2 \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Policy quantization similarity functions" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "Squared Euclidean" "Correlation" "Manhattan" "Euclidean" \
    --legend_at "upper left" \
    $*
