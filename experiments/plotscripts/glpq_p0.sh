#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset109 $LOGBASE/preset110  \
    --mode sweep \
    --trace_by lpq_p0 \
    --image_suffix policy_quantization_best \
    --no_log_scale \
    --log_scale \
    --xrange 0.8 0.95 \
    --yrange 0 1 \
    --title "Initial maximum policy value" \
    --xlabel "$ p(t=0) $" \
    --ylabel "Mean score" \
    --labels "A3C GLPQ" " " \
    --legend_at "upper left" \
    $*
