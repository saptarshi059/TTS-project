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

    for dataset_name in ["2wikimultihopqa", "hotpotqa", "musique"]:
        print(f"Collecting all contexts for {dataset_name}...")
        all_contexts = []

        dataset = pd.read_json(f"../../sampled_data/{dataset_name}/sampled_ds.json")
        all_contexts.extend(collect_contexts(dataset, dataset_name))

        context_lens = pd.Series([len(x) for x in tokenizer(all_contexts)['input_ids']])
        min_size, max_size, avg_tokens = context_lens.min(), context_lens.max(), context_lens.mean()
        print(f"Min, Max, Average tokens for {dataset_name}: {round(min_size, 2)}, {round(max_size, 2)},"
              f" {round(avg_tokens, 2)}")

if __name__ == "__main__":
    main()