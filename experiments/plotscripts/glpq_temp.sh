#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset126 $LOGBASE/preset94 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "GLPQ temp schedules" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "GLPQ hot" "GLPQ cold" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*

#  $LOGBASE/preset94 $LOGBASE/preset103 $LOGBASE/preset106 $LOGBASE/preset104 \
# "GLPQ $\gamma(t)$ $\eta(\gamma)$" "GLPQ $\gamma(t)$ $\eta(\gamma)$ NG" "GLPQ TT" "GLPQ NB"