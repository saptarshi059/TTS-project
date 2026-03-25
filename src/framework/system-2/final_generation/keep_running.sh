#!/bin/bash

DATASETS=("2wikimultihopqa")

for DS in "${DATASETS[@]}"; do
    echo "Starting processing for dataset: $DS"


    python ../../../utils/answer_scorer.py --prediction_dataset_path "../../../../framework_output/$DS/system2/final_responses.jsonl" --predicted_answer_field "final_ans" --ground_truth_field "answer"

    echo "Finished processing $DS successfully."
    echo "-----------------------------------"
done