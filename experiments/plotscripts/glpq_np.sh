#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset13 $LOGBASE/preset112 \
                $LOGBASE/preset113 $LOGBASE/preset94 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix glpq_np \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "GLPQ Prototypes Per Class" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "A3C FF" "A3C GLPQ $ \mathcal P / \mathcal A=32$" \
    "A3C GLPQ $ \mathcal P / \mathcal A=64$" \
    "A3C GLPQ $ \mathcal P / \mathcal A = 16$" \
    --legend_at "upper left" \
    $*
