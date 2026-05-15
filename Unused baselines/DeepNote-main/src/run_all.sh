#!/bin/bash

python main.py --method deepnote --retrieve_top_k 5 --dataset "2wikimultihopqa" --max_step 1 --max_fail_step 1 --model "qwen2.5-7b-instruct" --device cuda:0
python main.py --method deepnote --retrieve_top_k 5 --dataset "hotpotqa" --max_step 1 --max_fail_step 1 --model "qwen2.5-7b-instruct" --device cuda:0
python main.py --method deepnote --retrieve_top_k 5 --dataset "musique" --max_step 1 --max_fail_step 1 --model "qwen2.5-7b-instruct" --device cuda:0