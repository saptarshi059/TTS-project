#!/bin/bash

# ===================== User setting ===================== #
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

BASE_MODEL=$1
LORA_PATH=$2
EXP_TYPE="agent"
PORT_BASE=8000
# REDUCED memory to leave room for Retriever and Agent
GPU_MEMORY_UTILIZATION=0.2
MAX_LORA_RANK=64
N=1  # Keep it simple for stability

# Use ONLY GPU 0 for everything
TARGET_GPU="0"

RETRIEVER_CONDA_ENV="retriever"
RETRIEVER_LOG="retriever_server.log"
# ===================================================== #

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

trap 'cleanup; exit 1' SIGINT SIGTERM

# 0. Run retriever server on GPU 0
echo "🔍 Launching retriever on GPU $TARGET_GPU..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_CONDA_ENV"

CUDA_VISIBLE_DEVICES=$TARGET_GPU \
  python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
conda deactivate
sleep 30

# 1. Start ONLY ONE vLLM instance on GPU 0
echo "🚀 Starting vLLM on GPU $TARGET_GPU..."
LOG_FILE="vllm_gpu0.log"

CUDA_VISIBLE_DEVICES=$TARGET_GPU python serve_vllm.py \
  --model "$BASE_MODEL" \
  --port $PORT_BASE \
  --max-num-seqs 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --enable-lora \
  --lora-modules finetune=$LORA_PATH \
  --max-lora-rank $MAX_LORA_RANK \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION > "$LOG_FILE" 2>&1 &

# 2. Wait for startup
echo "📺 Waiting for vLLM..."
( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
  if [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ vLLM Ready!"
    break
  fi
done

# 3. Run reasoning
# Use NO multithreading to ensure GPU 0 doesn't get overwhelmed
python -m exps_research.unified_framework.run_experiment \
  --experiment_type "$EXP_TYPE" \
  --data_path "data_processor/qa_dataset/test/hotpotqa_500_20250422.json" \
  --model_type vllm \
  --model_id "$BASE_MODEL" \
  --max_tokens 1024 \
  --n $N --temperature 0.4 --top_p 0.8 \
  --seed 42 \
  --fine_tuned --lora_folder "$LORA_PATH"

cleanup