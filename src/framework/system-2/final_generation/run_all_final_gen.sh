#!/bin/bash

DATASETS=("2wikimultihopqa")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python -u generate_with_all_evidence.py --dataset "$ds"
done