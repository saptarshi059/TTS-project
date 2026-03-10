from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import re
import sys
sys.path.append("../../../utils/")

from all_system_prompts import SYSTEM_2_MAIN_PROMPT


def main(dataset: str):
    base_path = Path(f"../../../../framework_output/system2/{dataset}/")

    system_1_generations = pd.read_json(f'../../../../framework_output/system1/{dataset}/system_1_complete.jsonl', lines=True)
    system_1_generations = system_1_generations.rename(columns={'cleaned_ans': 'final_ans'})
    system_1_generations = system_1_generations[['question', 'answer', 'final_ans']]

    streamed_responses = pd.read_json(base_path / "final_response/streamed_responses.jsonl", lines=True)
    main_dataset = pd.read_json(base_path / "retrieval_results/retrieved_docs.jsonl", lines=True)

    print('Parsing generations...')
    final_ans = []
    for row in streamed_responses.itertuples():
        split_ans = row.generation.split(SYSTEM_2_MAIN_PROMPT)[1].strip()
        match_obj = re.search(r'<final_answer>(.*?)</final_answer>', split_ans, re.DOTALL | re.IGNORECASE)
        try:
            final_ans.append(match_obj.group(1).strip())
        except:
            print(split_ans, "\n............")

    main_dataset['final_ans'] = final_ans
    main_dataset = main_dataset[['question', 'answer', 'final_ans']]

    print('Saving final dataset...')
    final_ds = pd.concat([main_dataset, system_1_generations])
    final_ds.to_json(base_path / "final_response/final_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)