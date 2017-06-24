#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset118 $LOGBASE/preset120 $LOGBASE/preset121 $LOGBASE/preset119 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix spatial_softmax \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Spatial Softmax on Catch" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C SS" "A3C SS TT" "A3C SS GT TT" "A3C WW" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
