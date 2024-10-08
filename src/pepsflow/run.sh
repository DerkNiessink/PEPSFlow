#!/bin/bash

# START PARAMS
# CURRENT PARAMETERS   
#===============================#
chi=4
d=2
lam=(2.8)
max_iter=50                     
runs=1
learning_rate=0.01              
epochs=30                       
perturbation=0.0    
gpu=false
save_folder=tests2
data_folder=None
#===============================#
# END PARAMS

save_fn="data/$save_folder/lam_$lam.pth"
data_fn="data/$data_folder/lam_$lam.pth"

echo "Running..."

jobs=()

if [ "$gpu" = true ]; then
    gpu_flag="--gpu"
else
    gpu_flag="--no-gpu"
fi

if [ "$data_folder" = "None" ]; then
    input_flag=""
else
    input_flag="--data_fn $data_fn"
fi

for l in "${lam[@]}"; do
    echo "Running for lam = $l"
 
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
            $gpu_flag"  # Include the gpu_flag here

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
    wait "$job"  # Wait for each job to complete
done

echo -e "\nAll tasks completed and data saved to 'data/$save_folder'."
