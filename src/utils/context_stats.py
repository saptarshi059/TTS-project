from collections import Counter

import pandas as pd
from transformers import AutoTokenizer


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


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    print("Collecting all contexts...")
    all_contexts = []
    for dataset_name in ["2wikimultihopqa", "hotpotqa", "musique"]:
        dataset = pd.read_json(f"../../sampled_data/{dataset_name}/sampled_ds.json")
        all_contexts.extend(collect_contexts(dataset, dataset_name))

    avg_tokens = pd.Series([len(x) for x in tokenizer(all_contexts)['input_ids']]).mean()
    print(avg_tokens)

if __name__ == "__main__":
    main()