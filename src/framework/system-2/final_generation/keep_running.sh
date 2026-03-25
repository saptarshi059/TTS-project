#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique" "frames")

for DS in "${DATASETS[@]}"; do
    echo "Starting processing for dataset: $DS"

    until python generate_with_all_evidence.py --dataset "$DS" --batch_size 8; do
        echo "Script crashed for $DS with exit code $?. Restarting..." >&2
        sleep 2
    done

    python parse_generations.py --dataset "$DS"

    echo "Finished processing $DS successfully."
    echo "-----------------------------------"
done