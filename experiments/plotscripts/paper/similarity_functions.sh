LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset98 $LOGBASE/preset99 $LOGBASE/preset100 \
    --mode sweep \
    --trace_by learning_rate \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Similarity functions" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "Manhattan"  "Euclidean" "Squared Euclidean" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
