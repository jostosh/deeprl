#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset125 $LOGBASE/preset124 \
    --mode sweep \
    --trace_by prototype_factor \
    --image_suffix policy_quantization_best \
    --no_log_scale \
    --xrange 0.5 10 \
    --yrange 0 1 \
    --title "GLPQ Prototype Factor" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "GLPQ single winner" "GLPQ collective winners" \
    --legend_at "upper left" \
    $*

#  $LOGBASE/preset94 $LOGBASE/preset103 $LOGBASE/preset106 $LOGBASE/preset104 \
# "GLPQ $\gamma(t)$ $\eta(\gamma)$" "GLPQ $\gamma(t)$ $\eta(\gamma)$ NG" "GLPQ TT" "GLPQ NB"