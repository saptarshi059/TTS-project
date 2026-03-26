from argparse import ArgumentParser
from pathlib import Path

import faiss
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import set_seed


def main(dataset: str):
    set_seed(42)

    print(f"Loading triple generation for {dataset}...")
    triple_generation_path = Path(f"../../../../framework_output/{dataset}/system2/triple_extraction/parsed_responses.jsonl")
    starting_ds = pd.read_json(triple_generation_path, lines=True)

    print(f"Loading FAISS index for {dataset} and moving to GPU...")
    index_path = f"../../../../sampled_data/{dataset}/{dataset}_index.index"

    dataset_index = faiss.read_index(index_path)

    print(f"Loading documents for {dataset} index...")
    doc_path = f"../../../../sampled_data/{dataset}/{dataset}-chunks.jsonl"
    dataset_docs = load_dataset('json', data_files=doc_path, split='train')

    print("Loading embedding model: Qwen3-Embedding-0.6B...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "dtype": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    flattened_triples_for_retrieval = []
    for row in starting_ds.itertuples():
        flattened_triples_for_retrieval.extend(row.generated_triples)

    print("Embedding generated triples for retrieval...")
    embedded_triples = model.encode(flattened_triples_for_retrieval, prompt_name="query", show_progress_bar=True)

    print("Performing semantic search...")
    _, doc_ids = dataset_index.search(embedded_triples, 5)

    print("Collecting the original documents...")
    all_retrieved_docs = []
    start_idx = 0
    for row in starting_ds.itertuples():
        num_triples = len(row.generated_triples)
        retrieved_ids = list(dict.fromkeys(doc_ids[start_idx: start_idx + num_triples].flatten()))[:20]
        retrieved_docs = [doc['contents'] for doc in dataset_docs.select(retrieved_ids)]
        all_retrieved_docs.append(retrieved_docs)
        start_idx = start_idx + num_triples

    print("Saving results...")
    starting_ds['retrieved_docs'] = all_retrieved_docs
    op_dir = Path(f"../../../../framework_output/{dataset}/system2/retrieval_results/")
    op_dir.mkdir(parents=True, exist_ok=True)
    starting_ds.to_json(op_dir / "with_retrieved_docs.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)