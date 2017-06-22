LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset96 $LOGBASE/preset97 $LOGBASE/preset98 $LOGBASE/preset99 $LOGBASE/preset100 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix rslpq_distances \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "RSLPQ distance functions" \
    --xlabel "$\log_{10}(\alpha)$" \
    --ylabel "Mean score" \
    --labels "Pearson" "Cosine" "Manhattan"  "Euclidean" "Squared Euclidean"  \
    --legend_at "upper left" \
    $*
