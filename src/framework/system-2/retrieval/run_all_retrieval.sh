#!/bin/bash

DATASETS=("2wikimultihopqa" "musique" "hotpotqa")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python retrieve_docs.py --dataset "$ds"
done