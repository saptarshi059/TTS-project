import os

from config import (
    DATA_PATH, HIPPORAG_CONFIG
)


def check_health():
    print(f"--- Service Health Check ---")
    print(f"Data Path Exists: {os.path.exists(DATA_PATH)}")
    print(f"LLM Endpoint: {HIPPORAG_CONFIG.get('llm_base_url')}")
    print(f"Embedding Model: {HIPPORAG_CONFIG.get('embedding_model_name')}")

    # Try a simple connection check to your LLM (the source of your 404)
    import httpx
    try:
        r = httpx.get(HIPPORAG_CONFIG.get('llm_base_url'))
        print(f"LLM Server Status: {r.status_code}")
    except Exception as e:
        print(f"LLM Server Unreachable: {e}")


if __name__ == "__main__":
    check_health()