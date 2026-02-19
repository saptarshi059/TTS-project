#!/bin/bash

ulimit -n 65535
ulimit -u 65535  # max user processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_USE_V1=0

# ===================== User setting ===================== #
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707"

BASE_DATA_DIR="../../../../sampled_data"
DATASET_NAME="2wikimultihopqa"

EXP_TYPE="agent"
PORT_BASE=13519
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=1
TEMP=0.4
MAX_TOKENS=1024

RETRIEVER_CONDA_ENV="retriever"
RETRIEVER_GPU_DEVICES="2,3"
RETRIEVER_LOG="retriever_server.log"
# ===================================================== #

declare -A DATASETS=(
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa.json"
)

PIDS=()

cleanup() {
  echo ""
  echo "🧹 Cleaning up servers..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done
  pkill -f 'vllm.entrypoints.openai.api_server'
  pgrep -f 'retriever_server.py' | xargs -r kill
  echo "✅ All servers stopped."
}

trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM

# ===================================================== #
# 0. Run retriever server
# ===================================================== #
echo "🔍 Launching retriever server..."
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
  RAYON_NUM_THREADS=1 \
  python search/retriever_server.py \
  --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
  --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
  --retriever_model "Qwen/Qwen3-Embedding-0.6B" \
  > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
PIDS+=($RETRIEVER_PID)

# ===================================================== #
# 1. Run 4 vLLM instances (one per GPU)
# ===================================================== #
NUM_GPUS=4
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CURRENT_PORT=$((PORT_BASE + i))
  LOG_FILE="vllm_gpu${i}.log"

  echo "🚀 Launching vLLM on GPU $i (API: $CURRENT_PORT)..."

  CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
      --model "$BASE_MODEL" \
      --port "$CURRENT_PORT" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len 8192 \
      --disable-log-requests \
      --trust-remote-code \
      --enable-lora \
      --lora-modules finetune="$LORA_PATH" \
      --max-lora-rank "$MAX_LORA_RANK" \
      --disable-frontend-multiprocessing > "$LOG_FILE" 2>&1 &

  PIDS+=($!)
done

# Wait for the LAST GPU to finish loading
LAST_LOG="vllm_gpu$((NUM_GPUS - 1)).log"
echo "⏳ Monitoring $LAST_LOG for startup..."
tail -n 0 -f "$LAST_LOG" | while read -r line; do
  echo "$line"
  if [[ "$line" == *"Uvicorn running on"* ]] || [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ All GPUs initialized!"
    pkill -P $$ tail
    break
  fi
done

# Extra: wait until all ports are actually responding
echo "⏳ Waiting for all vLLM servers to respond..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CURRENT_PORT=$((PORT_BASE + i))
  echo -n "  Checking GPU $i (port $CURRENT_PORT)..."
  until curl -s "http://localhost:$CURRENT_PORT/health" > /dev/null 2>&1; do
    sleep 2
  done
  echo " ✅"
done
echo "🟢 All vLLM servers are up!"

# ===================================================== #
# 2. Run experiment
# ===================================================== #
for dataset in "${!DATASETS[@]}"; do
  echo "🧠 Running reasoning on $dataset..."

  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --api_base \"http://localhost:$PORT_BASE/v1\" \
    --api_key \"token-abc\" \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --parallel_workers 1 \
    --use_process_pool \
    --n $N --temperature $TEMP --top_p 0.8 \
    --seed 42 \
    --verbose"

  if [ -n "$LORA_PATH" ]; then
    AGENT_CMD="$AGENT_CMD --fine_tuned --lora_folder \"$LORA_PATH\""
  fi

  eval $AGENT_CMD
done

RUN_EXIT_CODE=$?
cleanup
exit $RUN_EXIT_CODE