#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset74 $LOGBASE/preset28 $LOGBASE/preset27 $LOGBASE/preset95 \
    $LOGBASE/preset101 $LOGBASE/preset105 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix architectures \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Architectures" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C default" "A3C Conv LSTM" "A3C Spatial Softmax trainable $\tau$" "A3C Spatial Softmax static $\tau$" \
    "A3C SS global temp" "A3C WW" "A3C WW global temp" \
    --legend_at "upper left" \
    $*
