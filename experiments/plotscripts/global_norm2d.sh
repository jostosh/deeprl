#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset12 \
    --mode sweep \
    --trace_by learning_rate global_clip_norm \
    --image_suffix sweep_preset12 \
    --log_scale \
    --title "Global Clipping Norm Sweep" \
    --ylabel "$\log_{10}(c)$" \
    --xlabel "$\log_{10}(\eta)$" \
    --fontsize 20 \
    --display
