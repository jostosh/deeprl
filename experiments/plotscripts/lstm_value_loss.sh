#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset17 $LOGBASE/preset71 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix lstm_val_loss \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Value loss A3C LSTM" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "$\mathcal L_V = 1.0$" "$\mathcal L_V = 0.5$" \
    --legend_at "upper left" \
    $*
