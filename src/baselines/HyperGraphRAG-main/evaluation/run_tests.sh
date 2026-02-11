#!/bin/bash

# --- Ollama Setup ---
# Set the path to your local binary
OLLAMA_BIN="./ollama/bin/ollama"

# Check if Ollama is already running
if ! pgrep -f "$OLLAMA_BIN serve" > /dev/null
then
    echo "Starting Ollama from local binary..."
    # Start Ollama in the background and redirect logs
    $OLLAMA_BIN serve > ollama_server.log 2>&1 &

    # Wait for the server to initialize (adjust time if needed)
    sleep 15
else
    echo "Ollama is already running."
fi

# --- Step1. Knowledge HyperGraph Construction ---
python script_insert.py --cls 2wikimultihopqa
python script_insert.py --cls hotpotqa
python script_insert.py --cls musique

# --- Step2. Retrieve Knowledge of HyperGraphRAG ---
python script_hypergraphrag.py --data_source 2wikimultihopqa
python script_hypergraphrag.py --data_source hotpotqa
python script_hypergraphrag.py --data_source musique

# --- Step3. Generate Based on Retrieved Knowledge ---
python get_generation.py --data_sources 2wikimultihopqa --methods HyperGraphRAG
python get_generation.py --data_sources hotpotqa --methods HyperGraphRAG
python get_generation.py --data_sources musique --methods HyperGraphRAG

# --- Step4. Evaluate the Generation ---
CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source 2wikimultihopqa --method HyperGraphRAG
CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source hotpotqa --method HyperGraphRAG
CUDA_VISIBLE_DEVICES=0 python get_score.py --data_source musique --method HyperGraphRAG

# Kill Ollama after the script finishes to free up cluster resources
pkill -f "$OLLAMA_BIN serve"