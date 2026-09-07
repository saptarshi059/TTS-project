#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Update the model paths in the config.yaml file

# Run the actual tests
#python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset 2wikimultihopqa --max_step 3 --model "llama-3.1-8B-Instruct"
#python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --model "llama-3.1-8B-Instruct"
#python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset musique --max_step 3 --model "llama-3.1-8B-Instruct"
#python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset frames --max_step 3 --model qwen2.5-7b-instruct

python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset 2wikimultihopqa --max_step 3 --model "olmo-3-7B-Instruct"
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --model "olmo-3-7B-Instruct"
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset musique --max_step 3 --model "olmo-3-7B-Instruct"
