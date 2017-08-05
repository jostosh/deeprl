#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset112 \
                $LOGBASE/preset113 $LOGBASE/preset94 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix glpq_np \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Prototypes Per Action" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "$ |\mathcal W| / |\mathcal A|=32$" \
    "$ |\mathcal W| / |\mathcal A|=64$" \
    "$ |\mathcal W| / |\mathcal A| = 16$" \
    --legend_at "upper left" \
    --fontsize 22 \
    $*
