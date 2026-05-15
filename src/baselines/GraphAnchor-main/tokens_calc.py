import pandas as pd
from transformers import AutoTokenizer

two_wiki_reasoning = pd.read_json('output/2wikimultihopqa/emb/GraphAnchor/qwen2.5-7b-instruct/'
                                  'topk-5__max_step-3__max_fail_step-2-20260216-20:23:52.jsonl',
                                  lines=True).dropna(subset='reasoning_log')['reasoning_log'].tolist()
hp_ds_reasoning = pd.read_json('output/hotpotqa/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3__'
                               'max_fail_step-2-20260216-21:08:32.jsonl',
                               lines=True).dropna(subset='reasoning_log')['reasoning_log'].tolist()
mus_ds_reasoning = pd.read_json('output/musique/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3'
                                '__max_fail_step-2-20260216-21:53:39.jsonl',
                                lines=True).dropna(subset='reasoning_log')['reasoning_log'].tolist()

print(len(two_wiki_reasoning), len(hp_ds_reasoning), len(mus_ds_reasoning))




