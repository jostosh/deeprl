#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset115 $LOGBASE/preset116 $LOGBASE/preset117 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Local weight sharing" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "LWS PF 48" "LWS PF 64" "LWS PF 64 M" \
    --legend_at "upper left" \
    --display \
    --fontsize 20 \
    $*

#  $LOGBASE/preset94 $LOGBASE/preset103 $LOGBASE/preset106 $LOGBASE/preset104 \
# "GLPQ $\gamma(t)$ $\eta(\gamma)$" "GLPQ $\gamma(t)$ $\eta(\gamma)$ NG" "GLPQ TT" "GLPQ NB"