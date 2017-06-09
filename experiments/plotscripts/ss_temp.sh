#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset84 ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset84 \
    --mode sweep \
    --trace_by ss_temp \
    --image_suffix ss_temperature \
    --no_log_scale \
    --title "Spatial Softmax Temperature" \
    --xlabel "$\gamma_{SS}$" \
    --ylabel "Mean Score" \
    --labels "SS" " " \
    --xrange 0 2 \
    --yrange 0 1 \
    --display