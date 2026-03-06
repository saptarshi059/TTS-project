#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique" "frames")

for ds in "${DATASETS[@]}";do
  echo "-------------Running system-1 for ${ds}-------------"
  python triple_gen.py --dataset "$ds"
done