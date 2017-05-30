#!/usr/bin/env bash
python3 export_plots.py \
    --interpolate \
    --input_dir /home/jos/tensorflowlogs/adience \
    --xlabel "Train step" \
    --ylabel "Accuracy" \
    --scalar_subset "Accuracy/Validation" \
    --trace_by "model" \
    --title "SISWS on Adience"