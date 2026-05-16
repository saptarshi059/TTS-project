import pandas as pd
import re
from transformers import AutoTokenizer
import numpy as np


base_path = Path(f"../../../../framework_output/{dataset}/system2")
ds = pd.read_json(base_path / "retrieval_results/with_retrieved_docs.jsonl", lines=True)

all_texts = ''
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

