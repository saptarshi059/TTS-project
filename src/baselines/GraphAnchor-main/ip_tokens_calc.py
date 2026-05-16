import pandas as pd
from transformers import AutoTokenizer
import numpy as np

two_wiki_reasoning = pd.read_json('output/2wikimultihopqa/emb/GraphAnchor/qwen2.5-7b-instruct/'
                                  'topk-5__max_step-3__max_fail_step-2-20260216-20:23:52.jsonl',
                                  lines=True)
hp_ds_reasoning = pd.read_json('output/hotpotqa/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3__'
                               'max_fail_step-2-20260216-21:08:32.jsonl',
                               lines=True)
mus_ds_reasoning = pd.read_json('output/musique/emb/GraphAnchor/qwen2.5-7b-instruct/topk-5__max_step-3'
                                '__max_fail_step-2-20260216-21:53:39.jsonl',
                                lines=True)

all_texts = []
def collect_contexts(df):
    temp = []
    for row in df.itertuples():
        if isinstance(row.retrieve_ref_log, list):
            for x in row.retrieve_ref_log:
                temp.extend([y['contents'] for y in x['retrieve_refs']])
    return temp

all_texts.extend(collect_contexts(two_wiki_reasoning))
all_texts.extend(collect_contexts(hp_ds_reasoning))
all_texts.extend(collect_contexts(mus_ds_reasoning))


tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")
