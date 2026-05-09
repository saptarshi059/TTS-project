#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique")

for ds in "${DATASETS[@]}";do
  echo "-------------Running retrieval for ${ds}-------------"
  python reason.py --dataset_name "$ds"
done