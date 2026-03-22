import statistics

import pandas as pd
from datasets import load_dataset

from math_verify import parse, verify

from pathlib import Path

# 1. Get the directory where THIS script is saved
# 2. Navigate relative to that directory
script_dir = Path(__file__).parent
data_path = script_dir / "../../all_output/HuggingFaceH4_MATH-500/cot/streamed_generations.jsonl"

# 3. Resolve it to an absolute path so there's no guesswork
output_file = pd.read_json(data_path.resolve(), lines=True)

base_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test').to_pandas()
responses = []
for base_row, response_row in zip(base_dataset.itertuples(), output_file.itertuples()):
    gold = parse(base_row.answer)
    answer = parse(response_row.generation)

    responses.append(verify(gold, answer))

print(f"Accuracy: {statistics.mean(responses)*100:.2f}%")