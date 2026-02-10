#!/bin/bash

# 2wiki
python script_insert.py --cls 2wikimultihopqa
python script_hypergraphrag.py --data_source 2wikimultihopqa
python get_generation.py --data_sources 2wikimultihopqa --methods HyperGraphRAG
CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source 2wikimultihopqa --method HyperGraphRAG