import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import sys
sys.path.append("../../utils/")
from all_system_prompts import SYSTEM_2, TRIPLE_GEN

datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']

all_texts = []
for ds in datasets:
    comp = pd.read_json(f'../../experiment_runs/thresholding_run/{ds}/system1/system_1_complete.jsonl', lines=True)
    full = pd.read_json(f'../../experiment_runs/main_framework_run/{ds}/system2/retrieval_results/with_retrieved_docs.jsonl', lines=True)
    merged = pd.merge(comp, full, how='left')

    for row in merged.itertuples():
        formatted_triple_gen = (f"{TRIPLE_GEN}\n"
                                f"<input>\n"
                                f"Question: {row.question}\n"
                                f"Answer: {row.system_1_guess}\n"
                                f"</input>"
                                )
        all_texts.append(formatted_triple_gen)

        generated_triples_string = ", ".join(f"({triple})" for triple in row.generated_triples)
        retrieved_evidences = "\n\n".join(dict.fromkeys(row.retrieved_docs))
        formatted_final_sample = (f"{SYSTEM_2}\n"
                                  f"<input>\n"
                                  f"Question: {row.question}\n"
                                  f"Initial Guess: {row.system_1_guess}\n"
                                  f"Initial Reasoning: {generated_triples_string}\n"
                                  f"Retrieved Context: {retrieved_evidences}\n"
                                  f"</input>")

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

