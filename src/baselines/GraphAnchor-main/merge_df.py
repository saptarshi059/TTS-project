import pandas as pd

file_paths = {
    '2wikimultihopqa': ['../../../sampled_data/2wikimultihopqa/sampled_ds.json',
                        'output/2wikimultihopqa/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3__max_fail_step-2-20260216-20:23:52.jsonl'],
    'hotpotqa': ['../../../sampled_data/hotpotqa/sampled_ds.json',
                 'output/hotpotqa/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3__max_fail_step-2-20260216-21:08:32.jsonl',
                 ],
    'musique': ['../../../sampled_data/musique/sampled_ds.json',
                'output/musique/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3__max_fail_step-2-20260216-21:53:39.jsonl'
                ]
}

for ds, fp in file_paths.items():
    print(f"Processing {ds}")
    base = pd.read_json(fp[0])
    pred = pd.read_json(fp[1], lines=True)
    merged = pd.merge(base, pred, on='question', how='left')
    merged.to_json(f'output/{ds}/merged_df.jsonl', lines=True, orient='records', index=False)

