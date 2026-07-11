from argparse import ArgumentParser
from pathlib import Path

import faiss
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import set_seed
from tqdm import tqdm


def main(dataset: str):
    set_seed(42)

    print(f"Loading system-1 for {dataset} to use initial answers directly for retrieval...")
    s1_path = Path(f"../../../../framework_output/{dataset}/system1/parsed_responses.jsonl")
    starting_ds = pd.read_json(s1_path, lines=True)

    print(f"Loading FAISS index for {dataset}...")
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

    init_answers = []
    for row in starting_ds.itertuples():
        init_answers.append(row.system_1_guess)

    print("Embedding initial answers for retrieval...")
    embedded_answers = model.encode(init_answers, prompt_name="query", show_progress_bar=True)

    print("Performing semantic search...")
    _, doc_ids = dataset_index.search(embedded_answers, 5)

    print("Collecting the original documents...")
    retrieved_docs = []
    for hit_list in tqdm(doc_ids):
        retrieved_docs.append([doc['contents'] for doc in dataset_docs.select(hit_list)])

    print("Saving results...")
    starting_ds['retrieved_docs'] = retrieved_docs
    op_dir = Path(f"../../../../framework_output/{dataset}/system2/retrieval_results/")
    op_dir.mkdir(parents=True, exist_ok=True)
    starting_ds.to_json(op_dir / "with_retrieved_docs.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)