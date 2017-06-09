#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --mode sweep \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset72 $LOGBASE/preset78 $LOGBASE/preset79 $LOGBASE/preset82 \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Policy quantization" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C PQ $|\mathcal P^*|/|\mathcal A| = 1$" \
    "A3C PQ $|\mathcal P^*|/|\mathcal A| = 10$" "$|\mathcal P^*|/|\mathcal A| = 15$" "A3C PQ SI" \
    --legend_at "upper left" \
    $*
