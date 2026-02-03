#!/bin/bash

# ===================== User setting ===================== #
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

BASE_MODEL=$1
LORA_PATH=$2
EXP_TYPE="agent"
PORT_BASE=8000
# REDUCED memory to leave room for Retriever and Agent
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=1  # Keep it simple for stability

# Use ONLY GPU 0 for everything
TARGET_GPU="0"

RETRIEVER_CONDA_ENV="retriever"
RETRIEVER_LOG="retriever_server.log"
# ===================================================== #

cleanup() {
  echo "🧹 Total System Cleanup..."
  fuser -k -9 8000/tcp 2022/tcp ${PORT_RETRIEVER}/tcp 2>/dev/null
  pkill -u $(whoami) -9 -f vllm
  pkill -u $(whoami) -9 -f retriever_server
  echo "✅ Clear."
}

trap 'cleanup; exit 1' SIGINT SIGTERM

# 0. Run retriever server on CPU to save GPU memory for vLLM
echo "🔍 Launching retriever on CPU (for stability)..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_CONDA_ENV"

# Use CUDA_VISIBLE_DEVICES="" to force CPU
CUDA_VISIBLE_DEVICES="" python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
conda deactivate

# Wait for Retriever to actually be port-ready
sleep 30

# 1. Start ONLY ONE vLLM instance on GPU 0
echo "🚀 Starting vLLM on GPU $TARGET_GPU..."
CUDA_VISIBLE_DEVICES=$TARGET_GPU python serve_vllm.py \
  --model "$BASE_MODEL" \
  --port $PORT_BASE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-lora-rank $MAX_LORA_RANK \
  --lora-modules finetune=$LORA_PATH > vllm_gpu0.log 2>&1 &

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
if ps -p $RETRIEVER_PID > /dev/null; then
   echo "🏃 Starting Experiment..."
   python -m exps_research.unified_framework.run_experiment \
     --experiment_type "$EXP_TYPE" \
     --data_path "data_processor/qa_dataset/test/hotpotqa_500_20250422.json" \
     --model_type vllm \
     --model_id "$BASE_MODEL" \
     --fine_tuned --lora_folder "$LORA_PATH"
else
   echo "❌ Error: Retriever failed to start or crashed early."
   cleanup
   exit 1
fi