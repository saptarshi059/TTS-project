import pandas as pd
import re
from transformers import AutoTokenizer
import numpy as np

def extract_excluding_answer(text):
    # Regex to match <answer>...</answer> and any surrounding whitespace
    # \s* handles any extra newlines or spaces around the tag
    cleaned_text = re.sub(r'\s*<answer>.*?</answer>\s*', '', text, flags=re.DOTALL)

    return cleaned_text.strip()

wiki_outline = pd.read_json('output_data/outline_2wikimultihopqa.jsonl', lines=True)['init_page'].tolist()
wiki_response = pd.read_json('output_data/2wikimultihopqa_responses.jsonl', lines=True)['gen_text'].tolist()
wiki_response_cleaned = [extract_excluding_answer(x) for x in wiki_response]

hotpotqa_outline = pd.read_json('output_data/outline_hotpotqa.jsonl', lines=True)['init_page'].tolist()
hotpotqa_response = pd.read_json('output_data/hotpotqa_responses.jsonl', lines=True)['gen_text'].tolist()
hotpotqa_response_cleaned = [extract_excluding_answer(x) for x in hotpotqa_response]

musique_outline = pd.read_json('output_data/outline_musique.jsonl', lines=True)['init_page'].tolist()
musique_response = pd.read_json('output_data/musique_responses.jsonl', lines=True)['gen_text'].tolist()
musique_response_cleaned = [extract_excluding_answer(x) for x in musique_response]

all_texts = wiki_outline + wiki_response_cleaned + hotpotqa_outline + hotpotqa_response_cleaned + musique_outline + musique_response_cleaned
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

