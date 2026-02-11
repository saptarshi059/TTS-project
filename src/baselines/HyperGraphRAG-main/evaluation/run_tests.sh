#!/bin/bash

# --- Configuration ---
OLLAMA_BIN="./ollama/bin/ollama"
LOG_FILE="full_pipeline.log"

# --- 1. Start Ollama ---
if ! pgrep -f "$OLLAMA_BIN serve" > /dev/null
then
    echo "[$(date)] Starting Ollama..."
    $OLLAMA_BIN serve > ollama_server.log 2>&1 &
    sleep 15
else
    echo "[$(date)] Ollama is already running."
fi

# --- 2. Knowledge HyperGraph Construction ---
python -u script_insert.py --cls 2wikimultihopqa
python -u script_insert.py --cls hotpotqa
python -u script_insert.py --cls musique

# --- 3. Retrieve Knowledge of HyperGraphRAG ---
python -u script_hypergraphrag.py --data_source 2wikimultihopqa
python -u script_hypergraphrag.py --data_source hotpotqa
python -u script_hypergraphrag.py --data_source musique

# --- 4. Generate Based on Retrieved Knowledge ---
python -u get_generation.py --data_sources 2wikimultihopqa --methods HyperGraphRAG
python -u get_generation.py --data_sources hotpotqa --methods HyperGraphRAG
python -u get_generation.py --data_sources musique --methods HyperGraphRAG

# --- 5. Evaluate the Generation ---
CUDA_VISIBLE_DEVICES=0 python -u get_score.py --data_source 2wikimultihopqa --method HyperGraphRAG
CUDA_VISIBLE_DEVICES=0 python -u get_score.py --data_source hotpotqa --method HyperGraphRAG
CUDA_VISIBLE_DEVICES=0 python -u get_score.py --data_source musique --method HyperGraphRAG

echo "[$(date)] Pipeline complete."

# Shut down Ollama when finished
pkill -f "$OLLAMA_BIN serve"