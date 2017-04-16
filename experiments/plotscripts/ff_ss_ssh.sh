#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset39 $LOGBASE/preset40 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix spatial_softmax \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Policy quantization vs. default A3C" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C FF SS hierarchical" "A3C FF SS" \
    --legend_at "upper left" \
    $*
