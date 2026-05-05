#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python triple_gen.py --dataset "$ds" --model_name "XXsongLALA/Qwen-2.5-7B-base-RAG-RL"
  python parse_raw_responses.py --dataset "$ds"
done