#!/bin/bash

for ds in 2wikimultihopqa hotpotqa musique; do
  ./part1.sh "$ds"
  ./part2.sh "$ds"
  ./part3.sh "$ds"
done