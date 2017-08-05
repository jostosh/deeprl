#!/usr/bin/env bash
LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset107 $LOGBASE/preset108 $LOGBASE/preset94 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix glpq_neighborhood_size \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Competition Size" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels  "$ k=8$" "$ k=4$" "$ k=16$" \
    --legend_at "upper left" \
    --fontsize 22 \
    $*
