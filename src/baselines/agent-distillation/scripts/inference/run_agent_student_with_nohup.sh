#!/bin/bash

# ===================== User setting ===================== #
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_CHORD_MAX_RETRIES=1

BASE_MODEL=$1
LORA_PATH=$2
EXP_TYPE="agent"
PORT_BASE=8000
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=8
TEMP=0.4
MAX_TOKENS=1024

RETRIEVER_CONDA_ENV="retriever"
RETRIEVER_GPU_DEVICES="2,3"
RETRIEVER_LOG="retriever_server.log"
# ===================================================== #

declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/test/hotpotqa_500_20250422.json"
  ["musique"]="data_processor/qa_dataset/test/musique_500_20250504.json"
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa_500_20250511.json"
)

PIDS=()

# Cleanup function
cleanup() {
  echo ""
  echo "🧹 Cleaning up servers..."
  # Kill specifically tracked PIDs
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done

  # Fail-safe cleanup for vLLM and Retriever
  ps -u $USER -o pid,command | grep 'serve_vllm.py' | grep -v grep | awk '{print $1}' | xargs -r kill
  pgrep -f 'retriever_server.py' | xargs -r kill

  echo "✅ Cleanup complete."
}

# Handle termination signals (Added SIGHUP for nohup compatibility)
trap 'echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM SIGHUP
export VLLM_USE_V1=0

# ===================================================== #
# 0. Run retriever server
# ===================================================== #
echo "🔍 Launching retriever server..."
# Use absolute path for conda if 'conda' command isn't found in nohup
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate "$RETRIEVER_CONDA_ENV"
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
    python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
PIDS+=($RETRIEVER_PID)
echo "🛰️ Retriever server started (PID: $RETRIEVER_PID)"
conda deactivate

# ===================================================== #
# 1. Run vLLM servers
# ===================================================== #
for i in 0 1 2 3; do
  LOG_FILE="vllm_gpu${i}.log"
  CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
    --model \"$BASE_MODEL\" \
    --port $((PORT_BASE + i)) \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

  if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
  fi

  eval "$CMD" > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
  echo "🚀 Started vLLM on GPU $i (port $((PORT_BASE + i)))"
done

# ===================================================== #
# 2. Wait for Startup (Polling Method)
# ===================================================== #
# We check GPU 3's log as the indicator for readiness
LAST_LOG="vllm_gpu3.log"
echo "⌛ Waiting for vLLM to initialize (checking $LAST_LOG)..."

MAX_WAIT=60 # 5 minutes (60 * 5s)
RETRY=0
while ! grep -q "Application startup complete." "$LAST_LOG"; do
  sleep 5
  ((RETRY++))
  if [ $RETRY -ge $MAX_WAIT ]; then
    echo "❌ ERROR: vLLM timed out during startup. Check $LAST_LOG"
    cleanup
    exit 1
  fi
done
echo "✅ vLLM fully started, launching reasoning agent!"

# ===================================================== #
# 3. Run Experiments
# ===================================================== #
for dataset in "${!DATASETS[@]}"; do
  echo "🧠 Running reasoning on dataset: $dataset"
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
    --use_process_pool \
    --n $N --temperature $TEMP --top_p 0.8 \
    --seed 42 \
    --verbose"

  if [ -n "$LORA_PATH" ]; then
    AGENT_CMD="$AGENT_CMD --fine_tuned --lora_folder \"$LORA_PATH\""
  fi

  eval "$AGENT_CMD"
done

RUN_EXIT_CODE=$?

# ===================================================== #
# 4. Finalize
# ===================================================== #
cleanup

if [ $RUN_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Agent script failed with exit code $RUN_EXIT_CODE"
  exit $RUN_EXIT_CODE
else
  echo "✅ All tasks completed successfully"
  exit 0
fi