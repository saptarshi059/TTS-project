#!/bin/bash

# --- Environment & Setup ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export NCCL_IGNORE_DISABLED_P2P=1
export OMP_NUM_THREADS=1
ulimit -n 65535

DATASET_NAME=$1

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

# Infer Answers (GPUs 0, 1)
CUDA_VISIBLE_DEVICES=0,1 python ../src/infer_page.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --input_file "output_data/new_outline_${DATASET_NAME}_page.jsonl" \
    --output_file "output_data/${DATASET_NAME}_responses.jsonl" \
    --batch_size 32

# --- Final Cleanup ---
echo "🎉 Pipeline finished successfully. Cleaning up..."
kill "$(jobs -p)" 2>/dev/null
cleanup