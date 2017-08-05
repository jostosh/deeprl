#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset40 $LOGBASE/preset73 $LOGBASE/preset95 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix spatial_softmax \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Spatial Softmax on Catch" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C SS" "A3C FF SS elu" "A3C SS global temp"  \
    --legend_at "upper left" \
    --fontsize 22 \
    $*
