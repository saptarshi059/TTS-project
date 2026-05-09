#!/bin/bash

DATASETS=("2wikimultihopqa" "hotpotqa" "musique")

for ds in "${DATASETS[@]}";do
  echo "-------------Running reason for ${ds}-------------"
  until python reason.py --dataset_name "$ds" --batch_size 8; do
        echo "Script crashed for $DS with exit code $?. Restarting..." >&2
        sleep 2
    done
done