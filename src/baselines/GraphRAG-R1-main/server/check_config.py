import os

from config import (
    HF_ENDPOINT, CUDA_VISIBLE_DEVICES,
    LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, LLM_BASE_URL,
    SAVE_DIR, DATA_PATH, HIPPORAG_CONFIG,
    SERVER_HOST, SERVER_PORT, LOG_LEVEL
)


def check_health():
    print(f"--- Service Health Check ---")
    print(f"Data Path Exists: {os.path.exists(DATA_PATH)}")
    print(f"LLM Endpoint: {LLM_BASE_URL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")

    # Try a simple connection check to your LLM (the source of your 404)
    import httpx
    try:
        r = httpx.get(LLM_BASE_URL)
        print(f"LLM Server Status: {r.status_code}")
    except Exception as e:
        print(f"LLM Server Unreachable: {e}")


if __name__ == "__main__":
    check_health()