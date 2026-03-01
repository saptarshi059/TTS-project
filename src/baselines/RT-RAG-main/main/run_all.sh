#!/bin/bash

# Configuration
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"  # Change to your preferred LLM
EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"
LLM_PORT=12345
EMBED_PORT=14432
PYTHON_SCRIPT="load_data.py" # <--- Update this to your filename

echo "🚀 Starting vLLM Servers..."

# 1. Start the LLM Server (Background)
# Adjusted GPU memory to leave room for the embedding model
vllm serve "$LLM_MODEL" \
    --port $LLM_PORT \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code > llm_server.log 2>&1 &
LLM_PID=$!

# 2. Start the Embedding Server (Background)
vllm serve "$EMBED_MODEL" \
    --port $EMBED_PORT \
    --task embed \
    --gpu-memory-utilization 0.1 \
    --trust-remote-code > embed_server.log 2>&1 &
EMBED_PID=$!

# Function to check if a server is ready
wait_for_server() {
    local port=$1
    local name=$2
    echo "⏳ Waiting for $name to be ready on port $port..."
    while ! curl -s "http://localhost:$port/health" > /dev/null; do
        sleep 5
    done
    echo "✅ $name is ONLINE."
}

# 3. Validate Health
wait_for_server $LLM_PORT "LLM Server"
wait_for_server $EMBED_PORT "Embedding Server"

echo "🏁 Both servers are ready. Starting Python script..."

# 4. Run the Python Script
python "$PYTHON_SCRIPT"

# 5. Cleanup (Optional: Kills the servers when the Python script finishes)
echo "Stopping servers..."
kill $LLM_PID $EMBED_PID