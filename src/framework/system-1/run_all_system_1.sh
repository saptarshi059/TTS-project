#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique" "frames")

for ds in "${DATASETS[@]}";do
  echo "-------------Running system-1 for ${ds}-------------"
  python step1.py --dataset "$ds"
  python parse_system1_raw_response.py --dataset "$ds"
  python prepare_dataset_for_system_2.py --dataset "$ds"
done