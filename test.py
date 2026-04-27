import pandas as pd
import json

ds = "2wikimultihopqa"
sample_ds = pd.read_json(f"sampled_data/{ds}/sampled_ds.json")
all_contexts = []
if ds in {"2wikimultihopqa", "hotpotqa"}:
    for row in sample_ds.itertuples():
        row_ctx = row.context
        for ctx in row_ctx:
            title = ctx[0]
            text = ctx[1]
            all_contexts.append(f"{title}\n{' '.join(text)}")
else:
    for row in sample_ds.itertuples():
        row_ctx = row.paragraphs
        for ctx in row_ctx:
            title = ctx["title"]
            text = ctx["paragraph_text"]
            all_contexts.append(f"{title}\n{text}")

with open(f"sampled_data/{ds}/{ds}-chunks.jsonl", 'r', encoding='utf-8') as f:
    chunks = [x['content'] for x in json.load(f)]

print(set(chunks) == set(all_contexts))