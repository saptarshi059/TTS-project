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
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=1
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

# 1. Start Retriever on GPU 0
echo "🔍 Launching retriever server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python search/retriever_server.py \
    --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
    --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
    --retriever_model "Qwen/Qwen3-Embedding-0.6B" > retriever_server.log 2>&1 &
PIDS+=($!)

# 2. Start vLLM on GPU 1
echo "🚀 Starting vLLM on GPU 1..."
LOG_FILE="vllm_gpu1.log"

CMD="CUDA_VISIBLE_DEVICES=1 python serve_vllm.py \
  --model \"$BASE_MODEL\" \
  --port $PORT \
  --enforce-eager \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

if [ -n "$LORA_PATH" ]; then
  CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
fi

eval $CMD > "$LOG_FILE" 2>&1 &
PIDS+=($!)

# 3. Wait for vLLM to be ready
echo "📺 Waiting for vLLM startup..."
( tail -f "$LOG_FILE" & ) | while read line; do
  echo "$line"
  if [[ "$line" == *"Application startup complete."* ]]; then
    pkill -P $$ tail # Kill the background tail
    break
  fi
done

# 4. Run Experiment
echo "🧠 Running reasoning for: $DATASET_NAME"
AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
  --experiment_type \"$EXP_TYPE\" \
  --data_path \"$DATA_PATH\" \
  --api_base "http://localhost:8000/v1" \
  --api_key "token-abc" \
  --model_type vllm \
  --model_id \"$BASE_MODEL\" \
  --max_tokens $MAX_TOKENS \
  --multithreading \
  --use_process_pool \
  --n $N --temperature $TEMP --top_p 0.8 \
  --parallel_workers 1 \
  --use_single_endpoint
  --seed 42"

[ -n "$LORA_PATH" ] && AGENT_CMD="$AGENT_CMD --fine_tuned --lora_folder \"$LORA_PATH\""

eval "$AGENT_CMD"
RUN_EXIT_CODE=$?

cleanup
exit $RUN_EXIT_CODE