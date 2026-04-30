"""
Standalone HippoRAG indexing script.
Run this once per dataset to pre-build knowledge graphs before starting the server.

Usage:
    python index_datasets.py                          # index all datasets
    python index_datasets.py --dataset 2wikimultihopqa
    python index_datasets.py --dataset hotpotqa
    python index_datasets.py --dataset musique
"""

import argparse
import logging
import os
import sys

import pandas as pd

from config import (
    HF_ENDPOINT, CUDA_VISIBLE_DEVICES,
    LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, LLM_BASE_URL,
    SAVE_DIR, HIPPORAG_CONFIG,
    LOG_LEVEL
)
from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# Each entry: base_data_dir relative to project root, sub-path, and type tag
# ---------------------------------------------------------------------------
DATASETS = {
    "2wikimultihopqa": {
        "data_path": "../../HyperGraphRAG-main/evaluation/contexts/2wikimultihopqa_contexts.json",
        "save_dir": os.path.join(SAVE_DIR, "2wikimultihopqa"),
    },
    "hotpotqa": {
        "data_path": "../../HyperGraphRAG-main/evaluation/contexts/hotpotqa_contexts.json",
        "save_dir": os.path.join(SAVE_DIR, "hotpotqa"),
    },
    "musique": {
        "data_path": "../../HyperGraphRAG-main/evaluation/contexts/musique_contexts.json",
        "save_dir": os.path.join(SAVE_DIR, "musique"),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_docs(data_path: str) -> list[str]:
    """Load and flatten documents from a dataset JSON file."""
    logger.info(f"Reading {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Since we're only using the gold contexts.
    docs = [
  {
    "title": "FIRST PASSAGE TITLE",
    "text": pd.read_json(data_path)[0].to_list()[:1],
    "idx": 0
  }]

    if not docs:
        raise ValueError(f"No documents found in {data_path}")

    logger.info(f"Loaded {len(docs):,} documents")
    return docs


def build_hipporag(save_dir: str) -> HippoRAG:
    """Instantiate HippoRAG pointed at a specific save directory."""
    config = BaseConfig(**HIPPORAG_CONFIG)
    return HippoRAG(
        global_config=config,
        save_dir=save_dir,
        llm_model_name=LLM_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        llm_base_url=LLM_BASE_URL,
    )


def index_dataset(name: str, cfg: dict) -> None:
    """Run indexing for a single dataset."""
    logger.info(f"{'='*60}")
    logger.info(f"Indexing dataset: {name}")
    logger.info(f"  data_path : {cfg['data_path']}")
    logger.info(f"  save_dir  : {cfg['save_dir']}")
    logger.info(f"{'='*60}")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    docs = load_docs(cfg["data_path"])
    hipporag = build_hipporag(cfg["save_dir"])

    logger.info("Starting knowledge graph construction — this may take a while...")
    hipporag.index(docs=docs)
    logger.info(f"Done indexing '{name}'. Index saved to: {cfg['save_dir']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-build HippoRAG knowledge graph indexes.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to index (default: all)",
    )
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    failed = []
    for name, cfg in targets.items():
        try:
            index_dataset(name, cfg)
        except Exception as e:
            logger.error(f"Failed to index '{name}': {e}", exc_info=True)
            failed.append(name)

    if failed:
        logger.error(f"Indexing failed for: {failed}")
        sys.exit(1)

    logger.info("All indexing completed successfully.")


if __name__ == "__main__":
    main()
