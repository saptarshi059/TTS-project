#!/bin/bash
set -ex

export RAG_SERVER_ADDR_DIR="/gpuhome/sks6765/TTS-project/src/baselines/ASearcher-main/rag_server"
export PORT=3241
ulimit -n 65535
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false

DATASET="2wikimultihopqa"

WORK_DIR="/gpuhome/sks6765/TTS-project/sampled_data/${DATASET}"

index_file="${WORK_DIR}/${DATASET}_index.index"
corpus_file="${WORK_DIR}/${DATASET}-chunks.jsonl"

retriever_name="qwen3-0.6b"
retriever_path="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"

python3 -u tools/local_retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu --port $PORT --save-address-to $RAG_SERVER_ADDR_DIR