#!/bin/bash

# --- Resource Limits ---
ulimit -n 65535

# --- GPU/vLLM Optimizations ---
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ===================== User setting ===================== #
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
LORA_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707/"
EXP_TYPE="agent"
PORT=8000
GPU_MEMORY_UTILIZATION=0.9 # Increased since 7B is small for 4 GPUs
MAX_LORA_RANK=64
N=1
TEMP=0.4
MAX_TOKENS=1024
BASE_DATA_DIR="../../../../sampled_data"
# ===================================================== #

PIDS=()

cleanup() {
    echo "🧹 Cleaning up..."
    kill ${PIDS[*]} 2>/dev/null
    # Kill any leftover vLLM or retriever processes
    pkill -f 'vllm.entrypoints.openai.api_server'
    pkill -f 'retriever_server.py'
    echo "✅ All servers stopped."
}

trap 'cleanup; exit 1' SIGINT SIGTERM

DATASET_NAME=$1
DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"

# 1. Start Retriever on GPU 0
# We use CUDA_VISIBLE_DEVICES=0 here specifically for the retriever
echo "🔍 Launching retriever server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python search/retriever_server.py \
    --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
    --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
    --retriever_model "Qwen/Qwen3-Embedding-0.6B" > retriever_server.log 2>&1 &
PIDS+=($!)

# 2. Start vLLM using ALL 4 GPUs (Tensor Parallelism)
echo "🚀 Launching vLLM with Tensor Parallelism = 4..."
LOG_FILE="vllm_server.log"

# We use all 4 GPUs for the LLM.
# Note: vLLM will manage the 4 GPUs internally via --tensor-parallel-size
CMD="python -m vllm.entrypoints.openai.api_server \
    --model \"$BASE_MODEL\" \
    --port $PORT \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code"

if [ -n "$LORA_PATH" ]; then
    # Enable LoRA support in vLLM
    CMD="$CMD --enable-lora --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
fi

eval $CMD > "$LOG_FILE" 2>&1 &
PIDS+=($!)

# 3. Wait for vLLM to be ready
echo "📺 Waiting for vLLM to initialize..."
timeout 300s grep -q "Application startup complete." <(tail -f "$LOG_FILE")

if [ $? -ne 0 ]; then
    echo "❌ vLLM failed to start within 5 minutes. Check $LOG_FILE"
    cleanup
    exit 1
fi

echo "✅ vLLM fully started, launching reasoning agent!"

# 4. Run Reasoning Agent
echo "🧠 Running reasoning..."
AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"$DATA_PATH\" \
    --api_base \"http://localhost:$PORT/v1\" \
    --api_key \"token-abc\" \
    --model_type vllm \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
    --use_process_pool \
    --n $N --temperature $TEMP --top_p 0.8 \
    --parallel_workers 8 \
    --seed 42"

if [ -n "$LORA_PATH" ]; then
    AGENT_CMD="$AGENT_CMD --fine_tuned --lora_folder \"$LORA_PATH\""
fi

eval $AGENT_CMD
RUN_EXIT_CODE=$?

# 5. Cleanup and Exit
cleanup
exit $RUN_EXIT_CODE