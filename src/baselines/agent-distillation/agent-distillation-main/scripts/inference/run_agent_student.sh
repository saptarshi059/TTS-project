#!/bin/bash

ulimit -n 65535
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_IGNORE_DISABLED_P2P=1


# ===================== User setting ===================== #
BASE_MODEL="/gpuhome/sks6765/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"
LORA_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707/"

BASE_DATA_DIR="../../../../sampled_data"
DATASET_NAME="2wikimultihopqa"

EXP_TYPE="agent"
PORT_BASE=13579
GPU_MEMORY_UTILIZATION=0.6
MAX_LORA_RANK=64
N=8
TEMP=0.4

MAX_TOKENS=1024

RETRIEVER_CONDA_ENV="retriever"          # retriever conda
RETRIEVER_GPU_DEVICES="2,3"              # retriever GPU
RETRIEVER_LOG="retriever_server.log"     # retriever path
# ===================================================== #

declare -A DATASETS=(
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa.json"
)

PIDS=()

# 종료 핸들러 정의
cleanup() {
  echo ""
  echo "🧹 Cleaning up vLLM servers..."
  kill ${PIDS[*]} 2>/dev/null
  # If the process is not cleaned well
  ps -u $USER -o pid,command | grep 'vllm serve' | grep -v grep | awk '{print $1}' | xargs kill

  pgrep -f 'retriever_server.py' | xargs -r kill
  wait
  echo "✅ All vLLM servers stopped."
}

# Ctrl-C 처리
trap 'echo ""; echo "❌ Interrupted!"; cleanup; exit 1' SIGINT SIGTERM
export VLLM_USE_V1=0

# ===================================================== #
# 0. Run retriever server (background)
# ===================================================== #
echo "🔍 Launching retriever server in Conda env \"$RETRIEVER_CONDA_ENV\" …"
# Conda initialization
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
  python search/retriever_server.py \
  --index_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}_index.index" \
  --corpus_path "${BASE_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}-chunks.jsonl" \
  --retriever_model "Qwen/Qwen3-Embedding-0.6B" \
  > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!
echo "🛰️  Retriever server started (PID: $RETRIEVER_PID, GPUs: $RETRIEVER_GPU_DEVICES)"

PIDS+=($RETRIEVER_PID)

NUM_GPUS=4
# 1. Loop to launch 4 independent OpenAI-compatible servers
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CURRENT_PORT=$((PORT_BASE + i))
  LOG_FILE="vllm_gpu${i}.log"

  echo "🚀 Launching vLLM OpenAI Server on GPU $i (Port: $CURRENT_PORT)..."

  # Constructing the command
  # Note: TP=1 because we are running 1 instance per GPU
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
      --max-lora-rank "$MAX_LORA_RANK" > "$LOG_FILE" 2>&1 &

  PIDS+=($!)
done

# 2. Wait until the LAST GPU (GPU 3) is ready
LAST_LOG="vllm_gpu$((NUM_GPUS - 1)).log"
echo "⏳ Monitoring $LAST_LOG for startup..."

# Tail the log and wait for the success message
tail -n 0 -f "$LAST_LOG" | while read -r line; do
  echo "$line"
  if [[ "$line" == *"Uvicorn running on"* ]] || [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ All 4 GPUs are ready! Launching reasoning agent..."
    # Kill the tail process to stop the loop
    pkill -P $$ tail
    break
  fi
done

for dataset in "${!DATASETS[@]}"; do
  # 4. run experiment
  echo "🧠 Running reasoning..."
  AGENT_CMD="python -m exps_research.unified_framework.run_experiment \
    --experiment_type \"$EXP_TYPE\" \
    --data_path \"${DATASETS[$dataset]}\" \
    --model_type vllm \
    --api_base "http://localhost:$PORT_BASE/v1" \
    --api_key "token-abc" \
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
    --debug \
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
