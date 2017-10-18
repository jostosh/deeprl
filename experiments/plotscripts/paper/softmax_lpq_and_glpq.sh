LOGBASE=~/tensorflowlogs/peregrine/v0.9.5/sweep

python3 export_plots.py \
    --input_dir $LOGBASE/preset131 $LOGBASE/preset100 $LOGBASE/preset13 \
    --mode sweep \
    --trace_by learning_rate \
    --image_suffix rslpq_distances \
    --log_scale \
    --xrange -6 -2 \
    --yrange 0 1 \
    --title "Softmax, LPQ and GLPQ" \
    --xlabel "$\log_{10}(\eta)$" \
    --ylabel "Mean score" \
    --labels "GLPQ" "LPQ" "Softmax" \
    --legend_at "upper left" \
    --fontsize 20 \
    $*
