#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset59 \
    --mode sweep \
    --trace_by ppa wpr \
    --image_suffix sweep_preset59 \
    --title "Policy Quantization Inv. Sq. Euclidean" \
    --xlabel "$\log_{10}(|\mathcal P|/|\mathcal A|)$" \
    --ylabel "$|\mathcal P^*|/|\mathcal P|$" \
    --display