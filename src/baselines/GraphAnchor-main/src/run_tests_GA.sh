#!/bin/bash

# Build index
CUDA_VISIBLE_DEVICES=2 python -u build_index/emb/index.py --dataset 2wikimultihopqa --model qwen3-Embedding-0.6B
CUDA_VISIBLE_DEVICES=2 python -u build_index/emb/index.py --dataset hotpotqa --model qwen3-Embedding-0.6B
CUDA_VISIBLE_DEVICES=2 python -u build_index/emb/index.py --dataset musique --model qwen3-Embedding-0.6B
CUDA_VISIBLE_DEVICES=2 python -u build_index/emb/index.py --dataset frames --model qwen3-Embedding-0.6B

# Run the actual tests
CUDA_VISIBLE_DEVICES=2 python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset 2wikimultihopqa --max_step 3 --model qwen2.5-7b-instruct
CUDA_VISIBLE_DEVICES=2 python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset hotpotqa --max_step 3 --model qwen2.5-7b-instruct
CUDA_VISIBLE_DEVICES=2 python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset musique --max_step 3 --model qwen2.5-7b-instruct
CUDA_VISIBLE_DEVICES=2 python -u GraphAnchor.py --method GraphAnchor --retrieve_top_k 5 --dataset frames --max_step 3 --model qwen2.5-7b-instruct
