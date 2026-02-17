#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Run the actual tests
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset 2wikimultihopqa --max_step 3 --model qwen2.5-7b-instruct
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --model qwen2.5-7b-instruct
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset musique --max_step 3 --model qwen2.5-7b-instruct
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset frames --max_step 3 --model qwen2.5-7b-instruct
