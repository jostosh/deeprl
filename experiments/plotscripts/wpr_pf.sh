#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset58 \
    --mode sweep \
    --trace_by prototype_factor wpr \
    --image_suffix prototype_factor_wpr \
    --title "Grid search PQ" \
    --xlabel "$\log_{10}(\alpha_{\mathcal P} / \alpha)$" \
    --ylabel "$\mathcal P^* / \mathcal P$" \
    --display