#!/bin/bash

# Prevents memory fragmentation (crucial when running 4 separate vLLM instances)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Optimizes communication for multi-GPU setups (even if running separate instances)
export NCCL_IGNORE_DISABLED_P2P=1

# Forces vLLM to use the faster Triton kernels for attention and LoRA
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Optional: If you see "Too many open files" errors during high multithreading
ulimit -n 65535

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

RETRIEVER_GPU_DEVICES="2,3"
RETRIEVER_LOG="retriever_server.log"
# ===================================================== #

# 1. Argument Handling
if [ -z "$1" ]; then
  echo "❌ Usage: $0 <dataset_name>"
  exit 1
fi

DATASET_NAME=$1
# Rigid pathing: Always expects BASE_DATA_DIR/name/sampled_ds.json
DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"

if [ ! -f "$DATA_PATH" ]; then
  echo "❌ Dataset file not found: $DATA_PATH"
  exit 1
fi

PIDS=()

cleanup() {
  echo -e "\n🧹 Cleaning up servers..."
  kill ${PIDS[*]} 2>/dev/null
  ps -u $USER -o pid,command | grep 'vllm serve' | grep -v grep | awk '{print $1}' | xargs kill 2>/dev/null
  pgrep -f 'retriever_server.py' | xargs -r kill 2>/dev/null
  echo "✅ Done."
}

trap 'cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0

# 2. Launch Retriever (Single Env)
echo "🔍 Launching retriever on GPUs: $RETRIEVER_GPU_DEVICES..."
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES python search/retriever_server.py --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" > "$RETRIEVER_LOG" 2>&1 &
PIDS+=($!)

# 3. vLLM Optimizations
# --enable-prefix-caching: Speeds up repeated context (great for agents)
# --enable-chunked-prefill: Better throughput for long prompts
# --max-num-batched-tokens: Optimizes batch processing size
VLLM_OPTS="--gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
           --enable-prefix-caching \
           --enable-chunked-prefill \
           --max-num-batched-tokens 32768"

# 4. Start vLLM Servers
for i in 0 1 2 3; do
  LOG="vllm_gpu${i}.log"
  CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
       --model \"$BASE_MODEL\" \
       --port $((PORT_BASE + i)) \
       $VLLM_OPTS"

  if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK --enable-lora"
  fi

  eval "$CMD" > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "🚀 Started vLLM on GPU $i (Port $((PORT_BASE + i)))"

  # Only wait for the last one to be fully ready before starting the agent
  if [ $i -eq 3 ]; then
    echo "📺 Waiting for final vLLM startup..."
    ( tail -n 0 -f "$LOG" & ) | while read line; do
      if [[ "$line" == *"Application startup complete."* ]]; then
        pkill -P $$ tail
        break
      fi
    done
  fi
done

# 5. Run Experiment
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

cleanup
exit $RUN_EXIT_CODE