#!/bin/bash

# Start 2 connections first - serve ollama in one & run the scripts in the other.

# 2wiki
python script_insert.py --cls 2wikimultihopqa
python script_hypergraphrag.py --data_source 2wikimultihopqa
python get_generation.py --data_sources 2wikimultihopqa --methods HyperGraphRAG
CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source 2wikimultihopqa --method HyperGraphRAG