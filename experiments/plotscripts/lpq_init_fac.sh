#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset143 \
    --mode sweep \
    --trace_by learning_rate lpq_init_fac \
    --image_suffix sweep_preset12 \
    --log_scale \
    --title "LPQ init bounds" \
    --ylabel "$\log_{10}(\rho_{LPQ})$" \
    --xlabel "$\log_{10}(\eta)$" \
    --fontsize 20 \
    --display
