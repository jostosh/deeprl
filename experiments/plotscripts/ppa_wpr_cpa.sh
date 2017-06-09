#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset67 \
    --mode sweep \
    --trace_by ppa wpr \
    --image_suffix sweep_preset67 \
    --title "Neighborhood size and prototypes per action" \
    --xlabel "$\log_{10}(|\tilde{\mathcal A}^k|/|\mathcal A|)$" \
    --ylabel "$|\tilde{\mathcal A}^k|/|\tilde{\mathcal A}|$" \
    --display \
    --levels 30