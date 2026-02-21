#!/bin/sh

# export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DATASET_NAME=$1



corpus_file=/workspace/PruneRAG/corpus/wiki-18.jsonl
index_file=/share/datasets/data_wiki_index_flat/e5_Flat.index

retriever_name=e5
retriever_path=/workspace/PruneRAG/models/e5-base-v2

python ./scripts/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu

CONFIG_PATH="./config/dataset_paths.json"

RETRIEVER_NAME="bge"
TOPK=5


MODEL_PATH="./models/llama-3.1-8b-instruct"

python -m pipelines.tree_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name $RETRIEVER_NAME \
    --retrieval_url "http://localhost:8000" \
    --data_path $CONFIG_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk $TOPK \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

