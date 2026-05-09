import os
from argparse import ArgumentParser
from pathlib import Path

import faiss
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(dataset_name: str) -> None:
    base_path = Path(f"../../../sampled_data/{dataset_name}")

    # Loading test questions for the dataset
    dataset = pd.read_json(base_path / "sampled_ds.json")
    all_questions = dataset["question"].to_list()

    print(f"Loading FAISS index for {dataset_name}...")
    index_path = base_path / f"{dataset_name}_index.index"
    dataset_index = faiss.read_index(str(index_path))

    print(f"Loading documents for {dataset_name} index...")
    doc_path = base_path / f"{dataset_name}-chunks.jsonl"
    dataset_docs = load_dataset('json', data_files=str(doc_path), split='train')

    print("Loading embedding model...")
    # Loading embedding model
    model = SentenceTransformer(model_name_or_path="Qwen/Qwen3-Embedding-0.6B",
                                model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto",
                                              "dtype": "auto"},
                                tokenizer_kwargs={"padding_side": "left"}
                                )

    print("Creating embeddings .....")
    query_embeddings = model.encode(all_questions, prompt_name="query", show_progress_bar=True)

    print("Performing semantic search...")
    _, doc_ids = dataset_index.search(query_embeddings, 5)

    retrieved_docs = []
    for hit_list in tqdm(doc_ids):
        retrieved_docs.append([doc['contents'] for doc in dataset_docs.select(hit_list)])

    dataset["retrieved_docs"] = retrieved_docs

    print("Saving dataset with retrieved triples...")
    dataset.to_json(f"{dataset_name}_with_retrieved_triples_from_naive_baseline.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    args = parser.parse_args()

    main(dataset_name=args.dataset_name)