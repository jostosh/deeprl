#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset86 $LOGBASE/preset87 $LOGBASE/preset90 $LOGBASE/preset91 $LOGBASE/preset88\
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Policy quantization" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C GLPQ" "A3C GLPQ$^+$" "A3C GLPQ$^+$ Pearson" "A3C GLPQ$^+$ NG NBA" "A3C GLPQ$^+$ NG" \
    --legend_at "upper left" \
    $*
