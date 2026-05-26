#!/bin/bash

DATASETS=("2wikimultihopqa")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python triple_gen.py --dataset "$ds"
  python parse_raw_responses.py --dataset "$ds"
done