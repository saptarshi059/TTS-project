import pandas as pd
from transformers import AutoTokenizer
import numpy as np

two_wiki_reasoning = pd.read_json('output/2wikimultihopqa/emb/GraphAnchor/qwen2.5-7b-instruct/'
                                  'topk-5__max_step-3__max_fail_step-2-20260216-20:23:52.jsonl',
                                  lines=True).dropna(subset='reasoning_log')['reasoning_log'].tolist()
hp_ds_reasoning = pd.read_json('output/hotpotqa/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3__'
                               'max_fail_step-2-20260216-21:08:32.jsonl',
                               lines=True).dropna(subset='reasoning_log')['reasoning_log'].tolist()
mus_ds_reasoning = pd.read_json('output/musique/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3'
                                '__max_fail_step-2-20260216-21:53:39.jsonl',
                                lines=True).dropna(subset='reasoning_log')['reasoning_log'].tolist()

two_wiki_reasoning_flat = [x[0]['reasoning'] for x in two_wiki_reasoning]
hp_ds_reasoning_flat = [x[0]['reasoning'] for x in hp_ds_reasoning]
mus_ds_reasoning_flat = [x[0]['reasoning'] for x in mus_ds_reasoning]

all_reasoning = two_wiki_reasoning_flat + hp_ds_reasoning_flat + mus_ds_reasoning_flat

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")
