#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset42 $LOGBASE/preset70 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix ff_val_loss \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Value loss A3C FF coefficient" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "$\mathcal L_V = 1.0$" "$\mathcal L_V = 0.5$" \
    --legend_at "upper left" \
    $*
