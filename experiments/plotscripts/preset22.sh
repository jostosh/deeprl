#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset22 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix sweep_preset22 \
    --log_scale \
    --yrange 0 1 \
    --title "SISWS static centroids on Catch" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score final 10 evals" \
    $*
