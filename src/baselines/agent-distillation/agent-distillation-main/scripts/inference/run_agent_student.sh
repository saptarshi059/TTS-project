#!/bin/bash

# --- Configuration ---
SESSION_NAME="research_env"
PORT_VLLM=8000
PORT_RETRIEVER=8005
GPU_MEMORY_UTILIZATION=0.7 # Increased slightly for stability
DATASET_NAME=$1
DATA_PATH="../../../../sampled_data/${DATASET_NAME}/sampled_ds.json"
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"

# 1. Check if tmux session already exists; if not, create it
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "🏗️ Creating new tmux session: $SESSION_NAME"
  tmux new-session -d -s "$SESSION_NAME" -n "infrastructure"

  # Window 0: Retriever (GPU 0)
  echo "🔍 Launching Retriever on GPU 0 (Window 0)..."
  tmux send-keys -t "$SESSION_NAME:0" "CUDA_VISIBLE_DEVICES=0 python search/retriever_server.py \
    --port $PORT_RETRIEVER \
    --index_path '../../../../sampled_data/${DATASET_NAME}/${DATASET_NAME}_index.index' \
    --corpus_path '../../../../sampled_data/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl' \
    --retriever_model 'Qwen/Qwen3-Embedding-0.6B'" C-m

  # Window 1: vLLM (GPU 1)
  echo "🚀 Launching vLLM on GPU 1 (Window 1)..."
  tmux new-window -t "$SESSION_NAME:1" -n "vllm"
  tmux send-keys -t "$SESSION_NAME:1" "CUDA_VISIBLE_DEVICES=1 python serve_vllm.py \
    --model '$BASE_MODEL' \
    --port $PORT_VLLM \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION" C-m

  echo "⏳ Waiting 30s for servers to warm up..."
  sleep 30
else
  echo "✅ Servers are already running in tmux session: $SESSION_NAME"
fi

# 2. Run the Experiment in the CURRENT terminal (so you see the output)
echo "🧠 Starting Experiment..."
python -m exps_research.unified_framework.run_experiment \
  --experiment_type "agent" \
  --data_path "$DATA_PATH" \
  --api_base "http://localhost:$PORT_VLLM/v1" \
  --api_key "token-abc" \
  --model_type vllm \
  --model_id "$BASE_MODEL" \
  --max_tokens 1024 \
  --multithreading \
  --use_process_pool \
  --n 1 \
  --temperature 0.4 \
  --parallel_workers 4 \
  --use_single_endpoint \
  --seed 42

# NOTE: We NO LONGER call cleanup here.
# The servers will stay alive in tmux for your next run.
echo "🏁 Experiment finished. Servers are still running in tmux session '$SESSION_NAME'."
echo "Use 'tmux attach -t $SESSION_NAME' to view them."