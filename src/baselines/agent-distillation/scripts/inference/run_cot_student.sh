#!/bin/bash

# ===================== User Setting ===================== #
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

BASE_MODEL=$1
LORA_PATH=$2
EXP_TYPE="reasoning"
PORT_BASE=8000
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64

declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/test/hotpotqa_500_20250422.json"
  ["musique"]="data_processor/qa_dataset/test/musique_500_20250504.json"
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa_500_20250511.json"
)
MAX_TOKENS=4096
# ===================================================== #

PIDS=()

# set end handler
cleanup() {
  echo "🧹 Total System Cleanup..."

  # 1. Kill by Port (Works without root for your own processes)
  # fuser -k is usually fine, but let's be more manual to be safe
  for port in 8000 8005; do
    PORT_PIDS=$(lsof -t -i:$port)
    if [ -n "$PORT_PIDS" ]; then
      echo "Killing processes on port $port..."
      kill -9 $PORT_PIDS 2>/dev/null
    fi
  done

  # 2. Target the specific "VLLM::EngineCore" string
  # We use pgrep with -u to ensure you only kill your own processes
  ENGINE_PIDS=$(pgrep -u $(whoami) -f "VLLM::EngineCore")
  if [ -n "$ENGINE_PIDS" ]; then
    echo "Terminating orphaned EngineCores: $ENGINE_PIDS"
    echo $ENGINE_PIDS | xargs kill -9 2>/dev/null
  fi

  # 3. Catch the "truncated" name just in case
  # Linux often truncates process names in the task structure
  pkill -u $(whoami) -9 "VLLM::EngineCor" 2>/dev/null

  # 4. Global wipe for any remaining python/vllm/retriever strings
  pkill -u $(whoami) -9 -f "vllm"
  pkill -u $(whoami) -9 -f "retriever_server"
  pkill -u $(whoami) -9 -f "serve_vllm"

  echo "✅ Cleanup finished (User mode)."
}

# Ctrl-C
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
# export VLLM_USE_V1=0

# 0. run retriever as background if

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

for dataset in "${!DATASETS[@]}"; do
  # 4. run experiment
  echo "🧠 Running reasoning..."
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --use_rag \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --parallel_workers 1 \
    --use_single_endpoint \
    --task_type "fact" \
    --n 1 --temperature 0.0 --top_p 0.8 \
    --seed 42 \
    --verbose"

  if [ -n "$LORA_PATH" ]; then
    AGENT_CMD="$AGENT_CMD --fine_tuned --lora_folder \"$LORA_PATH\""
  fi

  eval $AGENT_CMD
done

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
