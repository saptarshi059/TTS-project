#!/bin/bash

DATASETS=("2wikimultihopqa" "musique" "hotpotqa")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python triple_gen.py --dataset "$ds"  --model_name "mistralai/Mistral-7B-Instruct-v0.3"
  python parse_raw_responses.py --dataset "$ds"
done