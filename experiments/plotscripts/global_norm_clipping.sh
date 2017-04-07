#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset31 $LOGBASE/preset32 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix sweep_multiple_test \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Clipping analysis default A3C" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF clip by value" "A3C FF global norm clipping" "A3C ConvLSTM global norm clipping"  \
    --legend_at "upper left" \
    $*
