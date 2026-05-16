import pandas as pd
from transformers import AutoTokenizer
import numpy as np

datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']

all_texts = []
for ds in datasets:
    df = pd.read_json(f'{ds}_with_retrieved_triples_from_naive_baseline.json')
    for row in df.itertuples():
        all_texts.extend(row.retrieved_docs)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

