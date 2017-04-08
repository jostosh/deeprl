#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset36 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Policy quantization vs. default A3C" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C FF PQ" \
    --legend_at "upper left" \
    $*
