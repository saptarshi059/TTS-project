#!/bin/bash

# --- Resource Limits & Threading ---
ulimit -n 65535
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# --- GPU/vLLM Optimizations ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0

# ===================== User setting ===================== #
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
LORA_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707/"
EXP_TYPE="agent"
PORT=8000
GPU_MEMORY_UTILIZATION=0.85
MAX_LORA_RANK=64
N=8
TEMP=0.4
MAX_TOKENS=1024
BASE_DATA_DIR="../../../../sampled_data"
# ===================================================== #

PIDS=()

cleanup() {
  echo "🧹 Cleaning up..."
  kill ${PIDS[*]} 2>/dev/null
  pgrep -f 'vllm serve' | xargs -r kill -9
  pgrep -f 'retriever_server.py' | xargs -r kill -9
  echo "✅ All servers stopped."
}

trap 'cleanup; exit 1' SIGINT SIGTERM

DATASET_NAME=$1
DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"

echo $DATA_PATH