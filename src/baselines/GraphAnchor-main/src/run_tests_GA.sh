#!/bin/bash

# Build index
python build_index/emb/index.py --dataset 2wikimultihopqa --model bge-base-en-v1.5
python build_index/emb/index.py --dataset hotpotqa --model bge-base-en-v1.5
python build_index/emb/index.py --dataset musique --model bge-base-en-v1.5
python build_index/emb/index.py --dataset frames --model bge-base-en-v1.5

# Run the actual tests
CUDA_VISIBLE_DEVICES=2 python GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset 2wikimultihopqa --max_step 3 --model qwen2.5-7b-instruct
CUDA_VISIBLE_DEVICES=2 python GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --model qwen2.5-7b-instruct
CUDA_VISIBLE_DEVICES=2 python GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset musique --max_step 3 --model qwen2.5-7b-instruct
CUDA_VISIBLE_DEVICES=2 python GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset frames --max_step 3 --model qwen2.5-7b-instruct
