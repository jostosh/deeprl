#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset44 \
    --mode sweep \
    --trace_by ppa nwp \
    --image_suffix sweep_preset41 \
    --log_scale \
    --title "Policy Quantization Prototype GMM initialization" \
    --xlabel "$\log_{10}(|\mathcal P|/|\mathcal A|)$" \
    --ylabel "$\log_{10}(|\mathcal P^*|)$" \
    --display