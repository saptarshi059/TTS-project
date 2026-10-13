#!/bin/bash

for ds in 2wikimultihopqa hotpotqa musique; do
  ./run_prunerag.sh "$ds"
done