#!/bin/bash

# START PARAMS
# CURRENT PARAMETERS   
#===============================#
chi=32
d=3
lam=(3.0 3.05 3.1)
max_iter=10
runs=5
learning_rate=1.0
epochs=10
perturbation=0.0
gpu=false
threads=1
save_folder=test
data_folder=None
#===============================#
# END PARAMS


jobs=()

if [ "$gpu" = true ]; then
    gpu_flag="--gpu"
else
    gpu_flag="--no-gpu"
fi

for l in "${lam[@]}"; do
    echo "Optimizing for lam = $l"

    save_fn="data/$save_folder/lam_$l.pth"
    data_fn="data/$data_folder/lam_$l.pth"

    if [ "$data_folder" = "None" ]; then
        input_flag=""
    else
        input_flag="--data_fn $data_fn"
    fi
 
    cmd="python src/pepsflow/run.py \
            --chi \"$chi\" \
            --D \"$d\" \
            --lam \"$l\" \
            --max_iter \"$max_iter\" \
            --runs \"$runs\" \
            --lr \"$learning_rate\" \
            --epochs \"$epochs\" \
            --perturbation \"$perturbation\" \
            --save_fn \"$save_fn\" \
            --nThreads \"$nThreads\" \
            $gpu_flag"  

    # Only add input_flag if it is not empty
    if [ -n "$input_flag" ]; then
        cmd="$cmd $input_flag"
    fi

    (eval $cmd) &

    # Store the PID of the job
    jobs+=($!)  # $! gets the PID of the last background command
done

# Wait for all jobs to complete
for job in "${jobs[@]}"; do
    echo "$job"
    wait "$job"  # Wait for each job to complete
done

echo -e "\nAll tasks completed and data saved to '$save_folder'."
