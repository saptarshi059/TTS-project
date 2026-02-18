#!/bin/bash

# --- Environment & Performance Tuning ---
ulimit -n 65535
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_IGNORE_DISABLED_P2P=1

# ===================== User Settings ===================== #
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
LORA_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707/"
BASE_DATA_DIR="../../../../sampled_data"
DATASET_NAME=$1

# Inference Config
PORT=49152
GPU_UTIL=0.90      # High util since model is small vs 80GB A100
MAX_LORA_RANK=64
MAX_TOKENS=1024
WORKERS=32         # Increased concurrency for 500 samples
# ========================================================= #

PIDS=()
cleanup() {
    echo -e "\n🧹 Cleaning up processes..."
    [[ ${#PIDS[@]} -gt 0 ]] && kill "${PIDS[@]}" 2>/dev/null
    pkill -f 'vllm.entrypoints.openai.api_server'
    pkill -f 'retriever_server.py'
    echo "✅ Done."
}
trap 'cleanup; exit 1' SIGINT SIGTERM

DATA_PATH="${BASE_DATA_DIR}/${DATASET_NAME}/sampled_ds.json"

# 1. Start Retriever on CPU (Hidden from CUDA to save VRAM)
echo "🔍 Launching Retriever on CPU..."
CUDA_VISIBLE_DEVICES="" python search/retriever_server.py \
    --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
    --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
    --retriever_model "Qwen/Qwen3-Embedding-0.6B" > retriever.log 2>&1 &
PIDS+=($!)

# 2. Start vLLM on 1 GPU
# Note: --tensor-parallel-size 1 is faster for 7B models than splitting across 4 GPUs
echo "🚀 Launching vLLM on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    --disable-log-requests \
    --trust-remote-code \
    --enable-lora --lora-modules finetune="$LORA_PATH" --max-lora-rank $MAX_LORA_RANK > vllm.log 2>&1 &
PIDS+=($!)

# 3. Wait for Readiness
echo "📺 Waiting for vLLM startup..."
timeout 25s grep -q "Application startup complete." <(tail -f vllm.log) || { echo "❌ vLLM failed to start. Check vllm.log"; cleanup; exit 1; }

# 4. Run Experiment
echo "🧠 Running Reasoning Agent (Workers: $WORKERS)..."
python -m exps_research.unified_framework.run_experiment \
    --experiment_type "agent" \
    --data_path "$DATA_PATH" \
    --api_base "http://localhost:$PORT/v1" \
    --api_key "token-abc" \
    --model_type vllm \
    --model_id "$BASE_MODEL" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
    --use_process_pool \
    --debug \
    --parallel_workers $WORKERS \
    --n 1 --temperature 0.4 --top_p 0.8 --seed 42 \
    $( [[ -n "$LORA_PATH" ]] && echo "--fine_tuned --lora_folder $LORA_PATH" )

# 5. Finish
RUN_EXIT_CODE=$?
cleanup
exit $RUN_EXIT_CODE