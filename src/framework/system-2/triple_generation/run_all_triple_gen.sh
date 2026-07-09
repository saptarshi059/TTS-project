#!/bin/bash

DATASETS=("2wikimultihopqa" "musique" "hotpotqa")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python triple_gen.py --dataset "$ds"  --model_name "allenai/Olmo-3-7B-Instruct"
  python parse_raw_responses.py --dataset "$ds"
done