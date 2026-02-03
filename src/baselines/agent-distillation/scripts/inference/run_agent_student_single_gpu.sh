#!/bin/bash

# ===================== User setting ===================== #
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

BASE_MODEL=$1
LORA_PATH=$2
EXP_TYPE="agent"
PORT_BASE=8000
PORT_RETRIEVER=8005  # Matches your Python script
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=1
TARGET_GPU="0"
RETRIEVER_CONDA_ENV="retriever"
RETRIEVER_LOG="retriever_server.log"
VLLM_LOG="vllm_gpu0.log"
# ===================================================== #

cleanup() {
  echo "🧹 Total System Cleanup..."
  # Kill by ports
  fuser -k -9 ${PORT_BASE}/tcp ${PORT_RETRIEVER}/tcp 2>/dev/null
  # Kill by process name
  pkill -u $(whoami) -9 -f vllm
  pkill -u $(whoami) -9 -f retriever_server
  echo "✅ Clear."
}

trap 'cleanup; exit 1' SIGINT SIGTERM

# 0. Run retriever server on CPU
echo "🔍 Launching retriever on Port $PORT_RETRIEVER..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_CONDA_ENV"
CUDA_VISIBLE_DEVICES="" python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
conda deactivate

# Wait for Retriever Port to open
echo "⏳ Waiting for Retriever to bind to $PORT_RETRIEVER..."
timeout 60s bash -c "until q < /dev/tcp/localhost/$PORT_RETRIEVER; do sleep 2; done" 2>/dev/null

# 1. Start vLLM
echo "🚀 Starting vLLM on GPU $TARGET_GPU..."
CUDA_VISIBLE_DEVICES=$TARGET_GPU python serve_vllm.py \
  --model "$BASE_MODEL" \
  --port $PORT_BASE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-lora-rank $MAX_LORA_RANK \
  --lora-modules finetune=$LORA_PATH > "$VLLM_LOG" 2>&1 &

# 2. Wait for vLLM startup
echo "📺 Waiting for vLLM..."
timeout 300s grep -q "Application startup complete." <(tail -n 0 -f "$VLLM_LOG")
echo "✅ vLLM Ready!"

# 3. Run reasoning
if ps -p $RETRIEVER_PID > /dev/null; then
   echo "🏃 Starting Experiment..."
   # IMPORTANT: Ensure your experiment knows to look at PORT 8005
   python -m exps_research.unified_framework.run_experiment \
     --experiment_type "$EXP_TYPE" \
     --data_path "data_processor/qa_dataset/test/hotpotqa_500_20250422.json" \
     --model_type vllm \
     --model_id "$BASE_MODEL" \
     --parallel_workers 1 \
     --debug \
     --fine_tuned --lora_folder "$LORA_PATH"
     # If your experiment fails, add: --retriever_url http://localhost:8005
else
   echo "❌ Error: Retriever died."
   cleanup
   exit 1
fi

cleanup