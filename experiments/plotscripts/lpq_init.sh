#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset134 $LOGBASE/preset933 \
    --mode sweep \
    --trace_by lpq_p0 \
    --image_suffix policy_quantization_best \
    --log_scale \
    --no_log_scale \
    --xrange 0.5 0.95 \
    --yrange 0 1 \
    --title "LPQ starting temp" \
    --xlabel "$ p(t=0)$" \
    --ylabel "Mean score" \
    --labels "LPQ" " " \
    --legend_at "upper left" \
    --fontsize 20 \
    $*

#  $LOGBASE/preset94 $LOGBASE/preset103 $LOGBASE/preset106 $LOGBASE/preset104 \
# "GLPQ $\gamma(t)$ $\eta(\gamma)$" "GLPQ $\gamma(t)$ $\eta(\gamma)$ NG" "GLPQ TT" "GLPQ NB"