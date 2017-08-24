python3 export_plots.py --input_dir ~/tensorflowlogs/peregrine/v0.9.5/sweep/preset4 --mode sweep --trace_by learning_rate --image_suffix sweep_preset4 --title "Learning rate sweep with Elu on Catch" --yrange 0 1 --ylabel "Mean score final 10 evals" --xlabel "$\log_{10}(\alpha)$"