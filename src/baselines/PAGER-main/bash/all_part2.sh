#!/bin/bash
set -e

for ds in 2wikimultihopqa hotpotqa musique; do
  ./part2.sh "$ds"
done