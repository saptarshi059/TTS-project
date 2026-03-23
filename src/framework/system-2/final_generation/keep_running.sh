#!/bin/bash

DATASETS=("HuggingFaceH4/MATH-500")

for DS in "${DATASETS[@]}"; do
    echo "Starting processing for dataset: $DS"

    until python generate_with_all_evidence.py --dataset "$DS" --batch_size 1; do
        echo "Script crashed for $DS with exit code $?. Restarting..." >&2
        sleep 2
    done

    echo "Finished processing $DS successfully."
    echo "-----------------------------------"
done