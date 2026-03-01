#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

./run_dataset 2wikimultihopqa
./run_dataset hotpotqa
./run_dataset musique
./run_dataset frames