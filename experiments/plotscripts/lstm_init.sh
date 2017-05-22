#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset17 $LOGBASE/preset69 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix lstm_init \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "LSTM initialization" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C LSTM Torch init" "A3C LSTM TF init" \
    --legend_at "upper left" \
    $*
