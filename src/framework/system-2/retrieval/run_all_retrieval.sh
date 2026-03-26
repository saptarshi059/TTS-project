#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python retrieve_docs.py --dataset "$ds"
done