from datasets import load_dataset, tqdm
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

import sys
sys.path.append("../../utils/")

from all_system_prompts import *

def main(strategy, dataset):
    all_system_prompts = {'cot': COT, 'system1_math': SYSTEM_1_MATH}
    system_prompt = all_system_prompts[strategy]

    all_dataset_type = {"HuggingFaceH4/MATH-500": "math"}
    dataset_type = all_dataset_type[dataset]

    all_user_suffix = {'math': r"\n\nPlease put your final numerical or algebraic answer inside \boxed{}."}
    user_suffix = all_user_suffix[dataset_type]

    base_path = Path(f"../../../all_output/{dataset}/{strategy}")
    raw_responses = pd.read_json(base_path / "streamed_responses.jsonl", lines=True)
    ds = load_dataset(dataset, split='test').to_pandas()

    split_generations = []
    for base_row, response_row in zip(ds.itertuples(), raw_responses.itertuples()):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": rf"Question: {base_row.question} {user_suffix}"}]
        formatted_string = ""
        for element in messages:
            formatted_string += f"{element['role']}\n{element['content']}"
        formatted_string += "assistant"
        split_generations.append(response_row.generation.removeprefix(formatted_string).strip())

    ds['system_1_guess'] = split_generations
    ds.to_json(base_path / "system_1_split_generations.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--strategy", type=str, default='system1_math', choices=['system1_math', 'cot'])
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", choices=["HuggingFaceH4/MATH-500"])
    args = parser.parse_args()
    main(strategy=args.strategy, dataset=args.dataset)