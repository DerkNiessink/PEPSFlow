#!/bin/bash

# PARAMETERS
#==================================================================================================
chi=5
D=2
lamValues=(2.8)
max_iter=50
runs=5
learning_rate=0.01
epochs=30
perturbation=0.1
save_folder="tests"
input_folder="tests"
#==================================================================================================


echo "Running..."

jobs=()

for lam in "${lamValues[@]}"; do
    echo "Running for lam = $lam"
 
    (   pepsflow \
            --chi "$chi" \
            --D "$D" \
            --lam "$lam" \
            --max_iter "$max_iter" \
            --runs "$runs" \
            --lr "$learning_rate" \
            --epochs "$epochs" \
            --perturbation "$perturbation" \
            --save_fn "data/$input_folder/lam_$lam.pth" \
            #--data_fn "data/$input_folder/lam_$lam.pth" \
    ) &  # Run in the background

    # Store the PID of the job
    jobs+=($!)  # $! gets the PID of the last background command
done

# Wait for all jobs to complete
for job in "${jobs[@]}"; do
    wait "$job"  # Wait for each job to complete
done

echo "All tasks completed."
