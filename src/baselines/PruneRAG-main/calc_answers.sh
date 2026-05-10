#!/bin/bash

echo "2wikimultihopqa scores..."
python generate_combined_results.py --dataset "2wikimultihopqa"
python ../../utils/answer_scorer.py --prediction_dataset_path "outputs/2wikimultihopqa_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"

echo "Hotpotqa scores..."
python generate_combined_results.py --dataset "hotpotqa"
python ../../utils/answer_scorer.py --prediction_dataset_path "outputs/hotpotqa_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"

echo "Musique scores..."
python generate_combined_results.py --dataset "musique"
python ../../utils/answer_scorer.py --prediction_dataset_path "outputs/musique_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"