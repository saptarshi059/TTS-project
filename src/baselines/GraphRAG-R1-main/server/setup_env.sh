#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
export OPENAI_API_KEY="ollama"

# HippoRAG retrieval service environment setup script
# Usage: source setup_env.sh

echo "======================================"
echo "  HippoRAG Environment variables setup"
echo "======================================"

# ========== Required ==========
# ⚠️ Please set your API key below
export RERANK_API_KEY="${RERANK_API_KEY:-sk-your-api-key-here}"

# ========== Optional ==========

# HuggingFace mirror endpoint
export HF_ENDPOINT="https://huggingface.co"

# GPU device IDs (comma-separated, e.g. "0,1")
export CUDA_VISIBLE_DEVICES="0"

# LLM/OLLAMA settings

export OLLAMA_NUM_PARALLEL=16
export OLLAMA_MAX_LOADED_MODELS=4
export OLLAMA_PORT=10178
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

OLLAMA_BIN="./../../HyperGraphRAG-main/evaluation/ollama/bin/ollama"

if ! curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" > /dev/null 2>&1; then
    echo "[$(date)] Starting Ollama on port $OLLAMA_PORT..."
    OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" $OLLAMA_BIN serve > ollama_server.log 2>&1 &
    sleep 15
else
    echo "[$(date)] Ollama already running on port $OLLAMA_PORT."
fi

export LLM_MODEL_NAME="qwen2.5:7b-instruct" # Using basic instruct model for index building.
export LLM_BASE_URL="http://127.0.0.1:$OLLAMA_PORT/v1"

# Embedding model settings
export EMBEDDING_MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"

# Rerank model settings
export RERANK_BASE_URL="${RERANK_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export RERANK_MODEL="${RERANK_MODEL:-qwen-turbo-latest}"

# Data path (relative to project root)
export DATA_PATH="../../../../sampled_data/2wikimultihopqa/sampled_ds.json"

# Index save directory
export SAVE_DIR="${SAVE_DIR:-outputs/server}"

# Server settings
export SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
export SERVER_PORT="${SERVER_PORT:-8120}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

# ========== Confirmation ==========
echo ""
echo "已设置以下环境变量："
echo "  HF_ENDPOINT = $HF_ENDPOINT"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "  LLM_MODEL_NAME = $LLM_MODEL_NAME"
echo "  LLM_BASE_URL = $LLM_BASE_URL"
echo "  EMBEDDING_MODEL_NAME = $EMBEDDING_MODEL_NAME"
echo "  RERANK_API_KEY = ${RERANK_API_KEY:0:10}..." # show only first 10 chars
echo "  RERANK_BASE_URL = $RERANK_BASE_URL"
echo "  RERANK_MODEL = $RERANK_MODEL"
echo "  DATA_PATH = $DATA_PATH"
echo "  SAVE_DIR = $SAVE_DIR"
echo "  SERVER_HOST = $SERVER_HOST"
echo "  SERVER_PORT = $SERVER_PORT"
echo "  LOG_LEVEL = $LOG_LEVEL"
echo ""

# Validate required fields
if [ "$RERANK_API_KEY" = "sk-your-api-key-here" ] || [ -z "$RERANK_API_KEY" ]; then
    echo "⚠️  Warning: RERANK_API_KEY is not set or using default value"
    echo "   Please set RERANK_API_KEY in the script or via shell:"
    echo "   export RERANK_API_KEY='your-actual-api-key'"
    echo ""
fi

echo "======================================"
echo "Environment variables setup completed!"
echo ""
echo "Usage:"
echo "  source setup_env.sh    # load env vars"
echo "  python server.py       # start server"
echo "======================================"

# To just build the indices.
python index_datasets.py --dataset 2wikimultihopqa