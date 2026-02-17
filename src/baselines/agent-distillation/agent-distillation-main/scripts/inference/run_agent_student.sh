#!/bin/bash

# --- Resource Limits ---
ulimit -n 65535
ulimit -u 4096 2>/dev/null

# --- Threading Optimizations (Crucial for the OpenBLAS error) ---
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# --- GPU/vLLM Optimizations ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ===================== User setting ===================== #
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
LORA_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707/"
EXP_TYPE="agent"
PORT_BASE=8000
GPU_MEMORY_UTILIZATION=0.85  # Increased for better KV cache capacity
MAX_LORA_RANK=64
N=8
TEMP=0.4
MAX_TOKENS=1024

# Dataset Base Location
BASE_DATA_DIR="../../../../sampled_data"

#RETRIEVER_GPU_DEVICES="2,3"
RETRIEVER_LOG="retriever_server.log"
# ===================================================== #

PIDS=()

# 종료 핸들러 정의
cleanup() {
  echo ""
  echo "🧹 Cleaning up vLLM servers..."
  kill ${PIDS[*]} 2>/dev/null
  # If the process is not cleaned well
  ps -u $USER -o pid,command | grep 'vllm serve' | grep -v grep | awk '{print $1}' | xargs kill

  pgrep -f 'retriever_server.py' | xargs -r kill
  wait
  echo "✅ All vLLM servers stopped."
}

# Ctrl-C 처리
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0

DATASET_NAME=$1
# Rigid pathing: Always expects BASE_DATA_DIR/name/sampled_ds.json
DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"

# ===================================================== #
# 0. Run retriever server (background)
# ===================================================== #
echo "🔍 Launching retriever server..."
CUDA_VISIBLE_DEVICES=0 search/retriever_server.py --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" --retriever_model "Qwen/Qwen3-Embedding-0.6B" > "$RETRIEVER_LOG" 2>&1 &
PIDS+=($!)

# 1. GPU 0~2 background
for i in 0 1 2; do
  CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
    --model \"$BASE_MODEL\" \
    --port $((PORT_BASE + i)) \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

  if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
  fi

  eval $CMD > vllm_gpu${i}.log 2>&1 &
  PIDS+=($!)
  echo "🚀 Started vLLM on GPU $i (port $((PORT_BASE + i)))"
done

# 2. GPU 3 execute + log monitoring
i=3
LOG_FILE="vllm_gpu${i}.log"
CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
  --model \"$BASE_MODEL\" \
  --port $((PORT_BASE + i)) \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

if [ -n "$LORA_PATH" ]; then
  CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
fi

eval $CMD > "$LOG_FILE" 2>&1 &
PIDS+=($!)
echo "📺 Started final vLLM on GPU $i (port $((PORT_BASE + i))), watching for startup completion..."

# 3. wait until "Application startup complete." detected
( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
  echo "$line"
  if [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ vLLM fully started, launching reasoning agent!"
    break
  fi
done

echo "🧠 Running reasoning for: $DATASET_NAME"
AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
  --experiment_type \"$EXP_TYPE\" \
  --data_path \"$DATA_PATH\" \
  --model_type vllm \
  --model_id \"$BASE_MODEL\" \
  --max_tokens $MAX_TOKENS \
  --multithreading \
  --use_process_pool \
  --n $N --temperature $TEMP --top_p 0.8 \
  --seed 42 \
  --verbose"

[ -n "$LORA_PATH" ] && AGENT_CMD="$AGENT_CMD --fine_tuned --lora_folder \"$LORA_PATH\""

eval "$AGENT_CMD"
RUN_EXIT_CODE=$?

# 5. clean up server
cleanup

# 6. check exit code
if [ $RUN_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Agent script failed with exit code $RUN_EXIT_CODE"
  exit $RUN_EXIT_CODE
else
  echo "✅ Agent script completed successfully"
  exit 0
fi