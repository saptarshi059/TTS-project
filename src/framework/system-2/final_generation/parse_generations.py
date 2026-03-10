from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import re
import sys
sys.path.append("../../../utils/")

from all_system_prompts import SYSTEM_2_MAIN_PROMPT


def main(dataset: str):
    base_path = Path(f"../../../../framework_output/system2/{dataset}/")

    streamed_responses = pd.read_json(base_path / "final_response/streamed_responses.jsonl", lines=True)
    main_dataset = pd.read_json(base_path / "retrieval_results/retrieved_docs.jsonl", lines=True)

    print('Parsing generations...')
    final_ans = []
    for row in streamed_responses.itertuples():
        split_ans = row.generation.split(SYSTEM_2_MAIN_PROMPT)[1].strip()
        final_ans.append(re.search(r'<final_answer>(.*?)</final_answer>',
                                     split_ans, re.DOTALL | re.IGNORECASE)[1].strip())

    print('Saving final dataset...')
    main_dataset['final_ans'] = final_ans
    main_dataset.to_json(base_path / "final_response/parsed_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)