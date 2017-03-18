#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset12 \
    --mode sweep \
    --trace_by learning_rate global_clip_norm \
    --image_suffix sweep_preset12 \
    --log_scale \
    --title "Parameter sweep with Eve on Catch" \
    --yrange 0 1 \
    --ylabel "Mean score final 10 evals" \
    --xlabel "$\log_{10}(\alpha)$"
