#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

echo "-----------------Working on 2wikimultihopqa-----------------"
./run_dataset.sh 2wikimultihopqa

echo "-----------------Working on hotpotqa-----------------"
./run_dataset.sh hotpotqa

echo "-----------------Working on musique-----------------"
./run_dataset.sh musique

echo "-----------------Working on frames-----------------"
./run_dataset.sh frames