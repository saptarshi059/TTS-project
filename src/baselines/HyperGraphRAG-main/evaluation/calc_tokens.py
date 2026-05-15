import pandas as pd
import re
from transformers import AutoTokenizer
import numpy as np

def extract_content(text):
    # 1. Try to find content inside <think>...</think>
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    # 2. If <think> wasn't found, try to find content inside <answer>...</answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    # 3. If both failed, return the entire original text
    return text.strip()


wiki = pd.read_json('results/HyperGraphRAG/2wikimultihopqa/test_generation.json')['generation'].tolist()
hp = pd.read_json('results/HyperGraphRAG/hotpotqa/test_generation.json')['generation'].tolist()
mus = pd.read_json('results/HyperGraphRAG/musique/test_generation.json')['generation'].tolist()

wiki_cleaned = [extract_content(x) for x in wiki]
hp_cleaned = [extract_content(x) for x in hp]
mus_cleaned = [extract_content(x) for x in mus]

all_texts = wiki_cleaned + hp_cleaned + mus_cleaned
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

