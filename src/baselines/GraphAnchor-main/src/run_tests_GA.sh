#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Build index
python -u build_index/emb/index.py --dataset 2wikimultihopqa --model qwen3-Embedding-0.6B
python -u build_index/emb/index.py --dataset hotpotqa --model qwen3-Embedding-0.6B
python -u build_index/emb/index.py --dataset musique --model qwen3-Embedding-0.6B
python -u build_index/emb/index.py --dataset frames --model qwen3-Embedding-0.6B

# Run the actual tests
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset 2wikimultihopqa --max_step 3 --model qwen2.5-7b-instruct
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --model qwen2.5-7b-instruct
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset musique --max_step 3 --model qwen2.5-7b-instruct
python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset frames --max_step 3 --model qwen2.5-7b-instruct
