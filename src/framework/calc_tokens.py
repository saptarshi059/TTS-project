import pandas as pd
import re
from transformers import AutoTokenizer
import numpy as np


def extract_excluding_answer(text):
    # (?:final_)? allows it to match either 'answer' or 'final_answer'
    pattern = r'\s*<(?:final_)?answer>.*?</(?:final_)?answer>\s*'

    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

wiki_triples = pd.read_json('../../experiment_runs/main_framework_run/2wikimultihopqa/system2/triple_extraction/parsed_responses.jsonl', lines=True)['generated_triples'].tolist()
wiki_triples_flat = [item for sublist in wiki_triples for item in sublist]
wiki_responses = pd.read_json('../../experiment_runs/main_framework_run/2wikimultihopqa/system2/final_response/streamed_responses.jsonl', lines=True)['generation'].tolist()
wiki_responses_cleaned = [extract_excluding_answer(x) for x in wiki_responses]

hopotqa_triples = pd.read_json('../../experiment_runs/main_framework_run/hopotqa/system2/triple_extraction/parsed_responses.jsonl', lines=True)['generated_triples'].tolist()
hopotqa_triples_flat = [item for sublist in hopotqa_triples for item in sublist]
hopotqa_responses = pd.read_json('../../experiment_runs/main_framework_run/hopotqa/system2/final_response/streamed_responses.jsonl', lines=True)['generation'].tolist()
hopotqa_responses_cleaned = [extract_excluding_answer(x) for x in hopotqa_responses]

musique_triples = pd.read_json('../../experiment_runs/main_framework_run/musique/system2/triple_extraction/parsed_responses.jsonl', lines=True)['generated_triples'].tolist()
musique_triples_flat = [item for sublist in musique_triples for item in sublist]
musique_responses = pd.read_json('../../experiment_runs/main_framework_run/musique/system2/final_response/streamed_responses.jsonl', lines=True)['generation'].tolist()
musique_responses_cleaned = [extract_excluding_answer(x) for x in musique_responses]

all_texts = wiki_triples_flat + wiki_responses_cleaned + hopotqa_triples_flat + hopotqa_responses_cleaned + musique_triples_flat + musique_responses_cleaned
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

