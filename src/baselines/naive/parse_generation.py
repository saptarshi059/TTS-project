import re
import sys
from argparse import ArgumentParser

import pandas as pd

sys.path.append("../../utils/")

from all_system_prompts import NAIVE_BASELINE


def main(dataset: str):
    streamed_responses = pd.read_json(f"{dataset}_streamed_responses.jsonl", lines=True)
    main_dataset = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json")

    print('Parsing generations...')
    final_ans = []
    num_no_answer = 0
    for row in streamed_responses.itertuples():
        split_ans = row.generation.split(NAIVE_BASELINE)[1].strip()
        match_obj = re.search(r"\\+boxed\{([\s\S]*?)\}", split_ans, re.DOTALL | re.IGNORECASE)
        if match_obj:
            final_ans.append(match_obj.group(1).replace("\\", "").strip())
        else:
            num_no_answer += 1
            final_ans.append("No answer")

    print(f"Number of no_answers: {num_no_answer}...")

    streamed_responses['final_ans'] = final_ans
    final_ds = main_dataset.merge(streamed_responses, on=['question'])[['question', 'answer', 'final_ans']]

    print('Saving final dataset...')
    final_ds.to_json(f"{dataset}_final_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)