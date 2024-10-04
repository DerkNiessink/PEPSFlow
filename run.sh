#!/bin/bash

# Define the list of lam values
lam_values="0.5"

for lam in $lam_values; do
    python pepsflow/start_pepsflow.py \
        --chi 6 \
        --D 2 \
        --lam "$lam" \
        --max_iter 20 \
        --runs 1 \
        --lr 1 \
        --epochs 10 \
        --perturbation 0.0 \
        --fn "tests/test_lam_$lam" &
done

# Wait for all background processes to finish
wait

echo "All tasks completed."


