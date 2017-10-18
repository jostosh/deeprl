#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset86 $LOGBASE/preset93 $LOGBASE/preset94 $LOGBASE/preset106 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Temperature Configurations" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "$\tau=1$" "$\tau(t)$" \
    "$\tau(t), ~ \mathcal L_{\pi}(\tau)$" "$\tau$ trainable" \
    --legend_at "upper left" \
    --fontsize 22 \
    $*
