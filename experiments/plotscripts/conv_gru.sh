#!/usr/bin/env bash#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset35 $LOGBASE/preset38 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix conv_gru \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Conv GRU vs. FF" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C ConvGRU" "A3C ConvGRU no annealing" \
    --legend_at "upper left" \
    $*
