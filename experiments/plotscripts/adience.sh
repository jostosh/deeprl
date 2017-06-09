#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir /home/jos/tensorflowlogs/adience \
    --xlabel "Epoch" \
    --ylabel "Accuracy" \
    --scalar_subset "Accuracy/Validation" \
    --trace_by "model" \
    --title "SISWS on Adience" \
    --folds 3 \
    --xrange 0 200 \
    --export_best \
    $*