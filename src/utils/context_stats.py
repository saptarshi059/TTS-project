from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import Counter

def collect_contexts(dataset, dataset_name):
    all_contexts = []
    if dataset_name in {"2wikimultihopqa", "hotpotqa"}:
        for row in dataset.itertuples():
            row_ctx = row.context
            for ctx in row_ctx:
                title = ctx[0]
                text = ctx[1]
                all_contexts.append(f"{title}\n{' '.join(text)}")
    elif dataset_name == "musique":
        for row in dataset.itertuples():
            row_ctx = row.paragraphs
            for ctx in row_ctx:
                title = ctx["title"]
                text = ctx["paragraph_text"]
                all_contexts.append(f"{title}\n{text}")

    return all_contexts


def main(dataset_name):
    print("Loading embedding model...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                model_kwargs={"attn_implementation": "flash_attention_2",
                                              "device_map": "auto",
                                              "dtype": "auto"},
                                tokenizer_kwargs={"padding_side": "left"}
                                )
    tokenizer = model.tokenizer
    print("Embedding model loaded...")

    base_path = Path(f"../../sampled_data/{dataset_name}")
    dataset = pd.read_json(base_path / "sampled_ds.json")

    print("Collecting all contexts...")
    all_contexts = collect_contexts(dataset, dataset_name)

    token_dist = Counter([len(x) for x in tokenizer(all_contexts)['input_ids']])
    print(token_dist)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["2wikimultihopqa", "hotpotqa", "musique"])
    args = parser.parse_args()
    main(dataset_name=args.dataset)