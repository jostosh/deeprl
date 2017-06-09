#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --mode sweep \
    --input_dir $LOGBASE/preset72 $LOGBASE/preset78 $LOGBASE/preset79 $LOGBASE/preset82 \
    --trace_by learning_rate \
    --image_suffix policy_quantization_best \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "LPQ neighborhood size" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "$\frac{|\tilde{\mathcal A}^k|}{|\mathcal A|} = 1$" \
    "$\frac{|\tilde{\mathcal A}^k|}{|\mathcal A|} = 10$" "$\frac{|\tilde{\mathcal A}^k|}{|\mathcal A|} = 15$" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
