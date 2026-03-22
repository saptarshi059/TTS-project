import statistics

import pandas as pd
from datasets import load_dataset

from math_verify import parse, verify


gold = parse("${1,3} \\cup {2,4}$")
answer = parse("${1,2,3,4}$")

verify(gold, answer)

base_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test').to_pandas()
output_file = pd.read_json("../../all_output/HuggingFaceH4_MATH-500/cot/formatted_generations.jsonl", lines=True)
responses = []
for base_row, response_row in zip(base_dataset.itertuples(), output_file.itertuples()):
    responses.append(verify(base_row.answer, response_row.generation))

print(f"Accuracy: {statistics.mean(responses)*100:.2f}%")