#!/bin/bash

# --- Configuration ---
export CUDA_VISIBLE_DEVICES=3,4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export NCCL_IGNORE_DISABLED_P2P=1

# Threading limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

ulimit -n 65535

# --- Paths & Vars ---
BASE_DATA_DIR="../../../sampled_data"
DATASET_NAME=$1
DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"
RETRIEVER_LOG="./logs/retriever_${DATASET_NAME}.log"
MODEL_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

mkdir -p "./logs" "./outputs"

# --- THE TRAP ---
# This function runs automatically on script exit (success, error, or Ctrl+C)
cleanup() {
    echo -e "\n[CLEANUP] Shutting down retriever server (PID: $RETRIEVER_PID)..."
    kill "$RETRIEVER_PID" 2>/dev/null
    wait "$RETRIEVER_PID" 2>/dev/null
    echo "[CLEANUP] Done."
}
trap cleanup EXIT

# --- Start Retriever Server ---
# We use device 2 (physical GPU 5)
echo "[1/2] Starting retriever server on GPU 5..."
CUDA_VISIBLE_DEVICES=2 RAYON_NUM_THREADS=1 \
    python search/retriever_server.py \
        --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
        --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
        --retriever_model "Qwen/Qwen3-Embedding-0.6B" \
        >> "$RETRIEVER_LOG" 2>&1 &

RETRIEVER_PID=$!

# Wait for the server to be ready (Adjust sleep if your index is huge)
echo "Waiting 15s for retriever to initialize..."
sleep 15

# --- Run Pipeline ---
echo "[2/2] Launching tree pipeline..."
CUDA_VISIBLE_DEVICES=0,1 python -m pipelines.tree_pipeline \
    --model_path "$MODEL_PATH" \
    --retriever_name "qwen0.6b" \
    --retrieval_url "http://localhost:8005" \
    --data_path "$DATA_PATH" \
    --dataset_name "$DATASET_NAME" \
    --split "test" \
    --topk 5 \
    --max_depth 3 \
    --all_decom_depth 0 \
    --threshold 0.95 \
    --output_dir "./outputs" \
    --log_dir "./logs"

# The 'cleanup' function will now run automatically here because the script is exiting.