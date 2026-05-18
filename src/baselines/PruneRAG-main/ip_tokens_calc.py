import pandas as pd
from transformers import AutoTokenizer
import numpy as np

datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']
files = {
    '2wikimultihopqa': '2026-05-10 16:00:23.524448_rag_query_tree.jsonl',
    'hotpotqa': '2026-05-10 16:03:46.516218_rag_query_tree.jsonl',
    'musique': '2026-05-10 16:06:22.287385_rag_query_tree.jsonl'
}

all_texts = []
for ds in datasets:
    df = pd.read_json(f'logs/a09a35458c702b33eeacc393d103063234e8bc28/{ds}/{files[ds]}', lines=True)
    for row in df.itertuples():
        all_texts.extend([x[1] for x in row.context])

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

