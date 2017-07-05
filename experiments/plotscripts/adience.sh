#!/usr/bin/env bash
python3 export_plots.py \
    --input_dir /home/jos/tensorflowlogs/adience \
    --xlabel "Epoch" \
    --ylabel "Accuracy" \
    --scalar_subset "Accuracy/Validation" \
    --trace_by "model" \
    --title "LWS on Adience" \
    --folds 5 \
    --xrange 0 150 \
    --export_best \
    --step_to_epoch \
    --exclude per_feature \
    --fontsize 20 \
    $*