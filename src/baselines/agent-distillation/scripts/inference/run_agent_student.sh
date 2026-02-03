#!/bin/bash

# ===================== User setting ===================== #
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

BASE_MODEL=$1
LORA_PATH=$2 # set lora path here
EXP_TYPE="agent"
PORT_BASE=18000
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=1
TEMP=0.4

MAX_TOKENS=1024

RETRIEVER_CONDA_ENV="retriever"          # retriever conda
RETRIEVER_GPU_DEVICES="2,3"              # retriever GPU
RETRIEVER_LOG="retriever_server.log"     # retriever path
# ===================================================== #

declare -A DATASETS=(
  ["hotpotqa"]="data_processor/qa_dataset/test/hotpotqa_500_20250422.json"
  ["musique"]="data_processor/qa_dataset/test/musique_500_20250504.json"
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa_500_20250511.json"
)

PIDS=()

cleanup() {
  echo "🧹 Total System Cleanup..."

  # 1. Kill the tracked PIDs (the parents)
  [ ${#PIDS[@]} -gt 0 ] && kill ${PIDS[*]} 2>/dev/null

  # 2. Force-kill anything sitting on our specific ports
  # This is the most important part to avoid Errno 98
  for port in 8000 8001 8002 8003 8005; do
    echo "Clearing port $port..."
    fuser -k -9 ${port}/tcp 2>/dev/null
  done

  # 3. Final nuke for any stray vLLM/Python processes owned by you
  pkill -u $(whoami) -9 -f vllm
  pkill -u $(whoami) -9 -f retriever_server

  echo "✅ All ports and GPUs are now clear."
}

# Ctrl-C 처리
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0

# ===================================================== #
# 0. Run retriever server (background)
# ===================================================== #
# 0. Run retriever server
echo "🔍 Launching retriever server..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_CONDA_ENV"

# Use 'nohup' to prevent it from closing when the shell shifts focus
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
  nohup python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
PIDS+=($RETRIEVER_PID)

echo "🛰️  Retriever PID: $RETRIEVER_PID. Waiting for port 8005..."
# Wait specifically for the port to open
while ! nc -z localhost 8005; do
  sleep 1
done
echo "✅ Retriever is UP."
conda deactivate

# 1. Start all GPUs (0, 1, 2, and 3) in one clean loop
for i in 0 1 2 3; do
  VLLM_PORT=$((PORT_BASE + i))
  # These are the magic environment variables vLLM looks for:
  export MASTER_ADDR="127.0.0.1"
  export MASTER_PORT=$((10000 + i))  # Unique port for each GPU's internal comms
  export RAY_PORT=$((12000 + i))     # Unique Ray port

  echo "⏳ Launching vLLM on GPU $i (Port: $VLLM_PORT, Master Port: $MASTER_PORT)..."

  # Build the command
  CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
    --model \"$BASE_MODEL\" \
    --port $VLLM_PORT \
    --max-num-seqs 2 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --enforce-eager \
    --enable-lora \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

  if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
  fi

  # Execute and background
  eval $CMD > vllm_gpu${i}.log 2>&1 &
  PIDS+=($!)

  # If this is the LAST GPU (i=3), we wait for it to be fully ready
  if [ $i -eq 3 ]; then
    echo "📺 Watching GPU 3 logs for startup completion..."
    # This replaces your old 'tail' section
    ( tail -n 0 -f "vllm_gpu3.log" & ) | while read line; do
      echo "$line"
      if [[ "$line" == *"Application startup complete."* ]]; then
        echo "✅ All vLLM servers are UP! Launching Agent..."
        break
      fi
    done
  else
    # For GPUs 0, 1, and 2, just wait 20s before starting the next one
    sleep 20
  fi
done

for dataset in "${!DATASETS[@]}"; do
  # 4. run experiment
  echo "🧠 Running reasoning..."
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --n $N --temperature $TEMP --top_p 0.8 \
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
