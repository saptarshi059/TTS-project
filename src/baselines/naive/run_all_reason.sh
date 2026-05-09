#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique")

for ds in "${DATASETS[@]}";do
  echo "-------------Running reason for ${ds}-------------"
  until python reason.py --dataset_name "$ds" --batch_size 8; do
        echo "Script crashed for $DS with exit code $?. Restarting..." >&2
        sleep 2
    done

  python parse_generation.py --dataset "$ds"
  python ../../utils/answer_scorer.py --prediction_dataset_path "${ds}_final_responses.jsonl" --predicted_answer_field "final_ans" --ground_truth_field "answer"
done