#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset18 \
    --mode sweep \
    --trace_by fplc fp_decay \
    --image_suffix sweep_preset18 \
    --log_scale \
    --title "Sweep frame prediction" \
    --xlabel "$\log_{10}(\lambda_{FP})$" \
    --ylabel "$\log_{10}(\gamma_{FP})$" \
