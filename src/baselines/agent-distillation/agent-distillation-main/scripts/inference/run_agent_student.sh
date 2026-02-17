#!/bin/bash

# --- 1. Environment Fixes ---
# Prevents the script from trying to use a system proxy for local ports
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# --- 2. Configuration ---
SESSION_NAME="research_env"
PORT_VLLM=8000
PORT_RETRIEVER=8005
API_KEY="token-abc"

# Paths - Ensure these are absolute or correctly relative to where you run the script
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
LORA_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707/"
BASE_DATA_DIR="../../../../sampled_data"

DATASET_NAME=$1
if [ -z "$DATASET_NAME" ]; then
    echo "❌ Error: Please provide a dataset name (e.g., 2wikimultihopqa)"
    exit 1
fi

DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"

# --- 3. Persistent Infrastructure (tmux) ---
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "🏗️ Creating tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" -n "retriever"

    # Window 0: Retriever
    echo "🔍 Starting Retriever on GPU 0..."
    tmux send-keys -t "$SESSION_NAME:0" "CUDA_VISIBLE_DEVICES=0 python search/retriever_server.py \
        --port $PORT_RETRIEVER \
        --index_path '${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index' \
        --corpus_path '${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl' \
        --retriever_model 'Qwen/Qwen3-Embedding-0.6B'" C-m

    # Window 1: vLLM
    echo "🚀 Starting vLLM on GPU 1..."
    tmux new-window -t "$SESSION_NAME:1" -n "vllm"
    VLLM_CMD="CUDA_VISIBLE_DEVICES=1 python serve_vllm.py \
        --model '$BASE_MODEL' \
        --port $PORT_VLLM \
        --api-key $API_KEY \
        --max-model-len 8192 \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.7"

    if [ -n "$LORA_PATH" ]; then
        VLLM_CMD="$VLLM_CMD --lora-modules finetune=$LORA_PATH --max-lora-rank 64"
    fi

    tmux send-keys -t "$SESSION_NAME:1" "$VLLM_CMD" C-m
else
    echo "✅ Infrastructure already running in tmux session '$SESSION_NAME'"
fi

# --- 4. Health Check Loop ---
echo -n "⏳ Waiting for vLLM API to be ready (this can take 1-2 mins)"
MAX_RETRIES=60
COUNT=0
until curl -s -H "Authorization: Bearer $API_KEY" "http://localhost:$PORT_VLLM/v1/models" > /dev/null; do
    echo -n "."
    sleep 5
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo -e "\n❌ Timeout: vLLM failed to start. Check logs with: tmux attach -t $SESSION_NAME"
        exit 1
    fi
done
echo -e "\n✅ vLLM is ONLINE."

# --- 5. Run Experiment ---
echo "🧠 Running Experiment: $DATASET_NAME"
# Use the absolute path for model_id because that is how vLLM registered it
python -m exps_research.unified_framework.run_experiment \
  --experiment_type "agent" \
  --data_path "$DATA_PATH" \
  --api_base "http://localhost:$PORT_VLLM/v1" \
  --api_key "$API_KEY" \
  --model_type vllm \
  --model_id "$BASE_MODEL" \
  --max_tokens 1024 \
  --multithreading \
  --use_process_pool \
  --n 1 \
  --temperature 0.4 \
  --parallel_workers 4 \
  --use_single_endpoint \
  --seed 42 \
  --fine_tuned \
  --lora_folder "$LORA_PATH"

# End of script - Servers stay alive in tmux!
echo "🏁 Done. To stop servers later, run: tmux kill-session -t $SESSION_NAME"