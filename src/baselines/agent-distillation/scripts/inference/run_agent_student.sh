#!/bin/bash

# ===================== User setting ===================== #
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_CHORD_MAX_RETRIES=1

BASE_MODEL=$1
LORA_PATH=$2 # set lora path here
EXP_TYPE="agent"
PORT_BASE=8000
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
  ["hotpotqa"]="data_processor/qa_dataset/test/hotpotqa_500_20250422.json"
  ["musique"]="data_processor/qa_dataset/test/musique_500_20250504.json"
  ["2wiki"]="data_processor/qa_dataset/test/2wikimultihopqa_500_20250511.json"
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
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_CONDA_ENV"

# Use nohup to prevent SIGHUP and run directly in this shell
CUDA_VISIBLE_DEVICES=$RETRIEVER_GPU_DEVICES \
  nohup python search/retriever_server.py > "$RETRIEVER_LOG" 2>&1 &

RETRIEVER_PID=$!
PIDS+=($RETRIEVER_PID)
disown $RETRIEVER_PID

echo "🛰️  Retriever server started (PID: $RETRIEVER_PID)"
conda deactivate

PIDS+=($RETRIEVER_PID)

# 1. GPU 0~2 background
for i in 0 1 2; do
  CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
    --model \"$BASE_MODEL\" \
    --port $((PORT_BASE + i)) \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

  if [ -n "$LORA_PATH" ]; then
    CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
  fi

  eval $CMD > vllm_gpu${i}.log 2>&1 &
  PIDS+=($!)
  echo "🚀 Started vLLM on GPU $i (port $((PORT_BASE + i)))"
done

# 2. GPU 3 execute + log monitoring
i=3
LOG_FILE="vllm_gpu${i}.log"
CMD="CUDA_VISIBLE_DEVICES=$i python serve_vllm.py \
  --model \"$BASE_MODEL\" \
  --port $((PORT_BASE + i)) \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"

if [ -n "$LORA_PATH" ]; then
  CMD="$CMD --lora-modules finetune=$LORA_PATH --max-lora-rank $MAX_LORA_RANK"
fi

eval $CMD > "$LOG_FILE" 2>&1 &
PIDS+=($!)
echo "📺 Started final vLLM on GPU $i (port $((PORT_BASE + i))), watching for startup completion..."

# 3. wait until "Application startup complete." detected
( tail -n 0 -f "$LOG_FILE" & ) | while read line; do
  echo "$line"
  if [[ "$line" == *"Application startup complete."* ]]; then
    echo "✅ vLLM fully started, launching reasoning agent!"
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
    --model_id \"$BASE_MODEL\" \
    --max_tokens $MAX_TOKENS \
    --multithreading \
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
