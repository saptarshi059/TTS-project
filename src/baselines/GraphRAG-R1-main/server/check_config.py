import os

from config import (
    EMBEDDING_MODEL_NAME, LLM_BASE_URL,
    DATA_PATH
)


def check_health():
    print(f"--- Service Health Check ---")
    print(f"Data Path Exists: {os.path.exists(DATA_PATH)}")
    print(f"LLM Endpoint: {LLM_BASE_URL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

    # Try a simple connection check to your LLM (the source of your 404)
    import httpx
    try:
        # Check Ollama is alive
        r = httpx.get("http://127.0.0.1:10278/api/tags")
        print(f"Ollama reachable: {r.status_code}")

        # Check the OpenAI-compatible models endpoint
        r2 = httpx.get("http://127.0.0.1:10278/v1/models")
        print(f"v1/models status: {r2.status_code}")
        print(r2.json())
    except Exception as e:
        print(f"LLM Server Unreachable: {e}")


if __name__ == "__main__":
    check_health()