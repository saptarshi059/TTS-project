#!/bin/bash
set -ex

export RAG_SERVER_ADDR_DIR="127.0.0.1"
export PORT=3416

DATASET="2wikimultihopqa"

WORK_DIR="../../../../sampled_data/${DATASET}"

index_file="${WORK_DIR}/${DATASET}_index.index"
corpus_file="${WORK_DIR}/${DATASET}-chunks.jsonl"

retriever_name="qwen3-0.6b"
retriever_path="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B"

python3  tools/local_retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu --port $PORT --save-address-to $RAG_SERVER_ADDR_DIR