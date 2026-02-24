#!/bin/bash

echo "2wikimultihopqa scores..."
python ../../utils/answer_scorer.py --prediction_dataset_path "../baselines/PruneRAG-main/outputs/2wikimultihopqa_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"

echo "Hotpotqa scores..."
python ../../utils/answer_scorer.py --prediction_dataset_path "../baselines/PruneRAG-main/outputs/hotpotqa_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"

echo "Musique scores..."
python ../../utils/answer_scorer.py --prediction_dataset_path "../baselines/PruneRAG-main/outputs/musique_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"

echo "FRAMES scores..."
python ../../utils/answer_scorer.py --prediction_dataset_path "../baselines/PruneRAG-main/outputs/frames_parsed.jsonl" \
--predicted_answer_field "final_answer" --ground_truth_field "answer"