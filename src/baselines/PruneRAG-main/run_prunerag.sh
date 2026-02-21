#!/bin/sh

# export NCCL_P2P_DISABLE=1
ulimit -n 65535
export CUDA_VISIBLE_DEVICES=3,4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_USE_V1=0

BASE_DATA_DIR="../../../sampled_data"
DATASET_NAME=$1
DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"

RETRIEVER_GPU_DEVICES=5
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES RAYON_NUM_THREADS=1 \
    python search/retriever_server.py \
        --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
        --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
        --retriever_model "Qwen/Qwen3-Embedding-0.6B" \
        >> "$RETRIEVER_LOG" 2>&1

MODEL_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

python -m pipelines.tree_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name "qwen0.6b" \
    --retrieval_url "http://localhost:8005" \
    --data_path $DATA_PATH \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 5 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

