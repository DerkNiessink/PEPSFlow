#!/bin/bash

echo "Running..."

# Define the lam values
lamValues=(2.65 2.7 2.75 2.8 2.85 2.9 2.95 3.0 3.05 3.1 3.15 3.2)

# Create an array to hold job PIDs
jobs=()

# Iterate over each lam value
for lam in "${lamValues[@]}"; do
    echo "Running for lam = $lam"
    
    # Activate the virtual environment and run the Python script in the background
    ( 
        python src/pepsflow/start_pepsflow.py \
            --chi 6 \
            --D 2 \
            --lam "$lam" \
            --max_iter 20 \
            --runs 1 \
            --lr 1 \
            --epochs 10 \
            --perturbation 0.0 \
            --fn "tests/lam_$lam" \
            > "output/output_$lam.txt" 2>&1
    ) &  # Run in the background

    # Store the PID of the job
    jobs+=($!)  # $! gets the PID of the last background command
done

# Wait for all jobs to complete
for job in "${jobs[@]}"; do
    wait "$job"  # Wait for each job to complete
done

echo "All tasks completed."