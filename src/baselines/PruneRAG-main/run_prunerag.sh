#!/bin/bash

# --- Configuration ---

PIPELINE_GPUS="0"
RETRIEVER_GPU="1"
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
RETRIEVER_LOG="./logs/retriever_${DATASET_NAME}.log"
MODEL_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
RETRIEVER_URL="http://localhost:8005"

mkdir -p "./logs" "./outputs"

# --- THE TRAP ---
# This function runs automatically on script exit (success, error, or Ctrl+C)
cleanup() {
    echo -e "\n[CLEANUP] Killing retriever (PID: $RETRIEVER_PID)..."
    kill "$RETRIEVER_PID" 2>/dev/null
    wait "$RETRIEVER_PID" 2>/dev/null
}
trap cleanup EXIT

# --- Start Retriever ---
echo "[1/3] Starting retriever server on GPU 5..."
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU python scripts/search/retriever_server.py \
    --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
    --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
    --retriever_model "Qwen/Qwen3-Embedding-0.6B" \
    >> "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!

# --- Health Check Loop ---
echo "[2/3] Waiting for retriever to be healthy at $RETRIEVER_URL..."
MAX_RETRIES=30
RETRY_COUNT=0
# Loop until curl returns a successful exit code (0)
until curl -s "$RETRIEVER_URL" > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Retriever failed to start after $MAX_RETRIES seconds."
        exit 1
    fi
    # Check if the process died early
    if ! kill -0 $RETRIEVER_PID 2>/dev/null; then
        echo "ERROR: Retriever process crashed. Check $RETRIEVER_LOG"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo -e "\n[SUCCESS] Retriever is UP."

# --- Run Pipeline ---
#echo "[3/3] Launching tree pipeline..."
#CUDA_VISIBLE_DEVICES=$PIPELINE_GPUS python -m pipelines.tree_pipeline \
#    --model_path "$MODEL_PATH" \
#    --retriever_name "qwen0.6b" \
#    --retrieval_url $RETRIEVER_URL \
#    --dataset_name "$DATASET_NAME" \
#    --split "test" \
#    --topk 5 \
#    --max_depth 3 \
#    --all_decom_depth 0 \
#    --threshold 0.95 \
#    --output_dir "./outputs" \
#    --log_dir "./logs"

# The 'cleanup' function will now run automatically here because the script is exiting.
CUDA_VISIBLE_DEVICES=$PIPELINE_GPUS python -m pipelines.memorag_pipeline \
    --model_path $MODEL_PATH \
    --retriever_name "qwen0.6b" \
    --retrieval_url $RETRIEVER_URL \
    --dataset_name $DATASET_NAME \
    --split "test" \
    --topk 5 \
    --output_dir "./outputs" \
    --log_dir "./logs"