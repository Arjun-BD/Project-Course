#!/bin/bash

drop_edge_rates=(0.05 0.15 0.25 0.35 0.45 0.55)

log_file="experiment_results_drop_edge/experiment_results.log"

> $log_file

for rate in "${drop_edge_rates[@]}"; do
    echo "Running experiment with drop edge rate: $rate"

    output=$(python main.py --data_file=data/cora_ml --drop_random_edges True --privacy_amplify_sampling_rate 1 --pct_drop_random_edges $rate)

    accuracy=$(echo "$output" | grep -oP 'Testing accuracy: \K[0-9.]+')

    echo "Drop edge rate: $rate, Accuracy: $accuracy" >> $log_file
done

echo "Experiments completed. Results are logged in $log_file."