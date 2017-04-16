#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset41 \
    --mode sweep \
    --trace_by learning_rate prototype_factor \
    --image_suffix sweep_preset41 \
    --log_scale \
    --title "Policy Quantization" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "$\log_{10}(PF)$" \
    --display