#!/bin/bash

DATASETS=("2wikimultihopqa")

for DS in "${DATASETS[@]}"; do
    echo "Starting processing for dataset: $DS"

    until python generate_with_all_evidence.py --dataset "$DS" --batch_size 8 --gpu_id "1"; do
        echo "Script crashed for $DS with exit code $?. Restarting..." >&2
        sleep 2
    done

    echo "Finished processing $DS successfully."
    echo "-----------------------------------"
done