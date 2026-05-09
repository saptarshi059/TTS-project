#!/bin/bash

echo "2wikimultihopqa scores..."
python ../../utils/answer_scorer.py --prediction_dataset_path "outputs/2wikimultihopqa_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"

echo "Hotpotqa scores..."
python ../../utils/answer_scorer.py --prediction_dataset_path "outputs/hotpotqa_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"

echo "Musique scores..."
python ../../utils/answer_scorer.py --prediction_dataset_path "outputs/musique_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"