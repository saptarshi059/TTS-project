import statistics

import pandas as pd
from datasets import load_dataset

from math_verify import parse, verify


base_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test').to_pandas()
output_file = pd.read_json("/gpuhome/sks6765/TTS-project/all_output/HuggingFaceH4_MATH-500/cot/streamed_generations.jsonl", lines=True)
responses = []
for base_row, response_row in zip(base_dataset.itertuples(), output_file.itertuples()):
    gold = parse(base_row.answer)
    answer = parse(response_row.generation)

    responses.append(verify(gold, answer))

print(f"Accuracy: {statistics.mean(responses)*100:.2f}%")