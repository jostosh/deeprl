#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset57 $LOGBASE/preset65 $LOGBASE/preset62 $LOGBASE/preset72 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Policy quantization" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C PQ" "A3C PQ soft labels" "A3C PQ ZPI" "A3C PQ CPA" \
    --legend_at "upper left" \
    $*
