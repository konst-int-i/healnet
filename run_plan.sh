#!/bin/bash

# Check number of arguments
if [ "$#" -lt 1 ]; then
    echo "You need to specify at least one dataset."
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$#" -gt "$AVAILABLE_GPUS" ]; then
    echo "You have requested to run on $# GPUs but only $AVAILABLE_GPUS GPUs are available."
    exit 1
fi

# Run the command for each dataset
counter=0
for dataset in "$@"; do
    CUDA_VISIBLE_DEVICES=$counter python healnet/main.py --mode run_plan --dataset $dataset &
    counter=$((counter + 1))
done

# Wait for all processes to complete
wait