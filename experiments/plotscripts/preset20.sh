#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset20 \
    --mode sweep \
    --trace_by otc \
    --image_suffix sweep_preset20 \
    --log_scale \
    --yrange 0 1 \
    --title "Optimality tightening" \
    --xlabel "$\log_{10}(\lambda_{OT})$" \
    --ylabel "Mean score final 10 evals" \
    $*
