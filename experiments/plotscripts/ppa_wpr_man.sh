#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset52 \
    --mode sweep \
    --trace_by ppa wpr \
    --image_suffix sweep_preset52 \
    --title "Policy Quantization Manhattan distance" \
    --xlabel "$\log_{10}(|\mathcal P|/|\mathcal A|)$" \
    --ylabel "$|\mathcal P^*|/|\mathcal P|$" \
    --display