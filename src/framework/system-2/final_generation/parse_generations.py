from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import re
import sys
sys.path.append("../../../utils/")

from all_system_prompts import SYSTEM_2


def main(dataset: str):
    base_path = Path(f"../../../../framework_output/system2/{dataset}/")

    streamed_responses = pd.read_json(base_path / "final_response/streamed_responses.jsonl", lines=True)
    main_dataset = pd.read_json(base_path / "retrieval_results/with_retrieved_docs.jsonl", lines=True)

    print('Parsing generations...')
    final_ans = []
    num_no_answer = 0
    for row in streamed_responses.itertuples():
        split_ans = row.generation.split(SYSTEM_2)[1].strip()
        match_obj = re.search(r'<final_answer>(.*?)</final_answer>', split_ans, re.DOTALL | re.IGNORECASE)
        if match_obj:
            final_ans.append(match_obj.group(1).strip())
        else:
            num_no_answer += 1
            final_ans.append("No answer")

    print(f"Number of no_answers: {num_no_answer}...")

    main_dataset['final_ans'] = final_ans
    main_dataset = main_dataset[['question', 'system_1_guess', 'answer', 'final_ans']]

    print('Saving final dataset...')
    main_dataset.to_json(base_path / "final_response/final_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)