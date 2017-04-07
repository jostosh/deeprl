#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset33 \
    --mode sweep \
    --trace_by learning_rate ss_temp \
    --image_suffix sweep_preset33 \
    --log_scale \
    --title "Sweep Spatial Softmax Temperature" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "$\log_{10}(\tau_{SS})$" \
    --display