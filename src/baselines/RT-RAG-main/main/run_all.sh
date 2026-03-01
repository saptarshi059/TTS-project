#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

echo "-----------------Working on 2wikimultihopqa-----------------"
./run_dataset 2wikimultihopqa

echo "-----------------Working on hotpotqa-----------------"
./run_dataset hotpotqa

echo "-----------------Working on musique-----------------"
./run_dataset musique

echo "-----------------Working on frames-----------------"
./run_dataset frames