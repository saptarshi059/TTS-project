#!/bin/bash

DATASETS=("2wikimultihopqa" "musique" "hotpotqa")

for ds in "${DATASETS[@]}";do
  echo "-------------Running system-1 for ${ds}-------------"
  python step1.py --dataset "$ds"  --model_name "meta-llama/Llama-3.1-8B-Instruct"
  python parse_system1_raw_response.py --dataset "$ds"
  python prepare_dataset_for_system_2.py --dataset "$ds"
  python ../../utils/answer_scorer.py --prediction_dataset_path "../../../framework_output/${ds}/system1/parsed_responses.jsonl" --predicted_answer_field "system_1_guess" --ground_truth_field "gold_answer"
done