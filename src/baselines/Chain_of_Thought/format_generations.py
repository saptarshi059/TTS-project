from datasets import load_dataset, tqdm
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import sys
sys.path.append("../../utils/")

from all_system_prompts import COT

def main(dataset: str):
    responses_path = Path(f"../../../all_output/{dataset.replace('/','_')}/cot")

    raw_responses = pd.read_json(responses_path / "streamed_responses.jsonl", lines=True)
    base_dataset = load_dataset(dataset, split='test').to_pandas()

    stripped_generations = []
    for base_row, response_row in tqdm(zip(base_dataset.itertuples(), raw_responses.itertuples())):
        message = [{"role": "system", "content": COT},
                   {"role": "user", "content": rf"Question: {base_row.problem} \n\nPlease put your final numerical or algebraic answer inside \boxed{{}}."}]
        formatted_text = ""
        for element in message:
            formatted_text += f"{element['role']}\n{element['content']}\n"
        formatted_text += "assistant"

        stripped_generations.append(response_row.generation.strip(formatted_text))

    print("Saving formatted generations...")
    base_dataset['stripped_generation'] = stripped_generations
    base_dataset.to_json(responses_path / "formatted_generations.jsonl", lines=True, orient='records', index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", choices=["HuggingFaceH4/MATH-500"])
    args = parser.parse_args()
    main(dataset=args.dataset)