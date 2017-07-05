#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset122 $LOGBASE/preset999 \
    --mode sweep \
    --trace_by noiselevel \
    --image_suffix policy_quantization_best \
    --log_scale \
    --no_log_scale \
    --xrange 0 0.5 \
    --yrange 0 1 \
    --title "Noise Robustness" \
    --xlabel "Noise level" \
    --ylabel "Mean score" \
    --labels "A3C FF"  \
    --legend_at "upper left" \
    --fontsize 20 \
    $*

#  $LOGBASE/preset94 $LOGBASE/preset103 $LOGBASE/preset106 $LOGBASE/preset104 \
# "GLPQ $\gamma(t)$ $\eta(\gamma)$" "GLPQ $\gamma(t)$ $\eta(\gamma)$ NG" "GLPQ TT" "GLPQ NB"