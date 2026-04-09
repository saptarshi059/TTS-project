#!/bin/bash

DATASETS=("triviaqa")

for ds in "${DATASETS[@]}";do
  echo "-------------${ds}-------------"
  python retrieve_docs.py --dataset "$ds"
done