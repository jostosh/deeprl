#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset23 $LOGBASE/preset10 $LOGBASE/preset11 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix sweep_multiple_test \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Gradient descent optimizer comparison, default A3C" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C RMSProp" "A3C RMSEve" "A3C Adam" "A3C Eve"  \
    --legend_at "upper left" \
    $*
