#!/bin/bash

DATASETS=("2wikimultihopqa" "musique" "hotpotqa")

for DS in "${DATASETS[@]}"; do
    echo "Starting processing for dataset: $DS"

    until python generate_with_all_evidence.py --dataset "$DS" --batch_size 8 --model_name "meta-llama/Llama-3.1-8B"; do
        echo "Script crashed for $DS with exit code $?. Restarting..." >&2
        sleep 2
    done

    python parse_generations.py --dataset "$DS"
    python ../../../utils/answer_scorer.py --prediction_dataset_path "../../../../framework_output/${DS}/system2/final_response/final_responses.jsonl" --predicted_answer_field "final_ans" --ground_truth_field "gold_answer"

    echo "Finished processing $DS successfully."
    echo "-----------------------------------"
done