#!/bin/bash

DATASETS=("2wikimultihopqa")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python generate_with_all_evidence.py --dataset "$ds"
done