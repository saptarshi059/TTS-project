#!/bin/bash

# --- Environment & Setup ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export NCCL_IGNORE_DISABLED_P2P=1
export OMP_NUM_THREADS=1
ulimit -n 65535

DATASET_NAME=$1
BASE_DATA_PATH="../../../../sampled_data/${DATASET_NAME}"
DATA_PATH="${BASE_DATA_PATH}/sampled_ds.json"
INDEX_PATH="${BASE_DATA_PATH}/${DATASET_NAME}_index.index"
CORPUS_PATH="${BASE_DATA_PATH}/${DATASET_NAME}-chunks.jsonl"

EMBEDDING_PORT=65501
RETRIEVER_PORT=14325

# --- ZOMBIE PREVENTION: The Trap ---
# This ensures that if you Ctrl+C, the background servers are killed immediately.
cleanup() {
    echo -e "\n🛑 Interrupted! Shutting down background servers..."
    kill "$(jobs -p)" 2>/dev/null
    pkill -u "$(whoami)" -9 python
    pkill -u "$(whoami)" -9 VLLM
    exit 1
}
trap cleanup SIGINT SIGTERM

# --- Health Check Function ---
wait_for_server() {
    local url=$1
    local name=$2
    local search_term=$3
    echo "⏳ Waiting for $name..."
    # Loop until the server responds AND the search term is found
    while ! curl -s "$url" | grep -q "$search_term"; do
        sleep 5
    done
    echo "✅ $name is ONLINE."
}

# --- 1. Start Embedding Server (GPU 4) ---
# Running as a service because the Retriever script needs an API to talk to.
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen3-emb \
    --model "Qwen/Qwen3-Embedding-0.6B" \
    --trust-remote-code \
    --port $EMBEDDING_PORT \
    --runner pooling \
    --gpu-memory-utilization 0.55 &

wait_for_server "http://localhost:$EMBEDDING_PORT/v1/models" "vLLM Embedding" "qwen3-emb"

# --- 2. Start Retriever Server (GPU 5) ---
CUDA_VISIBLE_DEVICES=5 python ../src/retriever/ret_serve.py \
    --faiss_index_path "${INDEX_PATH}" \
    --corpus_jsonl_path "${CORPUS_PATH}" \
    --emb_url "http://localhost:${EMBEDDING_PORT}/v1" \
    --emb_model qwen3-emb \
    --gpu_ids 0 \
    --use_gpu True \
    --port $RETRIEVER_PORT &

wait_for_server "http://localhost:$RETRIEVER_PORT/health" "Retriever Service" "\"status\":\"ok\""

# Construct Outline (GPUs 2, 3)
CUDA_VISIBLE_DEVICES=2,3 python ../src/construct_outline.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --input_file "${DATA_PATH}" \
    --out_file "output_data/outline_${DATASET_NAME}.jsonl" \
    --max_iters 10 --batch_size 32 --seed 66 --resume

# Extract Outline (CPU Task)
python ../src/extract_outline.py \
    --json_file "output_data/outline_${DATASET_NAME}.jsonl" \
    --out_file "output_data/new_outline_${DATASET_NAME}.jsonl"


# --- Final Cleanup ---
echo "🎉 Pipeline finished successfully. Cleaning up..."
kill "$(jobs -p)" 2>/dev/null