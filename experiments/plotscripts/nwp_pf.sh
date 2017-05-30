#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset75 \
    --mode sweep \
    --trace_by prototype_factor nwp \
    --image_suffix nwp_pf \
    --title "Policy Quantization" \
    --xlabel "$\log_{10}(\alpha_{PQ}/\alpha)$" \
    --ylabel "$|\mathcal P^*|/|\mathcal A|$" \
    --display \
    --levels  30