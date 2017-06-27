#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset86 $LOGBASE/preset93 $LOGBASE/preset94 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "GLPQ temperature configurations" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "$\gamma=1$" "$\gamma(t)$" \
    "$\gamma(t), ~ \mathcal L_{\pi}(\gamma)$" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
