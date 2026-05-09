from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import re
import sys
sys.path.append("../../../utils/")

from all_system_prompts import SYSTEM_2, SYSTEM_2_ABLATION


def main(dataset: str, generation_mode:str):
    base_path = Path(f"../../../../framework_output/{dataset}")

    system_1_generations = pd.read_json(base_path/ "system1/system_1_complete.jsonl", lines=True)
    if not system_1_generations.empty:
        print("System 1 generations not empty...")
        system_1_generations = system_1_generations[['question', 'gold_answer', 'system_1_guess']]
        system_1_generations['final_ans'] = system_1_generations['system_1_guess']
    else:
        print("System 1 generations empty...")

    streamed_responses = pd.read_json(base_path / "system2/final_response/streamed_responses.jsonl", lines=True)
    main_dataset = pd.read_json(base_path / "system2/retrieval_results/with_retrieved_docs.jsonl", lines=True)[['question', 'gold_answer', 'system_1_guess']]

    print('Parsing generations...')
    final_ans = []
    num_no_answer = 0
    system_prompt = SYSTEM_2 if generation_mode == 'normal' else SYSTEM_2_ABLATION
    for row in streamed_responses.itertuples():
        split_ans = row.generation.split(system_prompt)[1].strip()
        match_obj = re.search(r'<final_answer>(.*?)</final_answer>', split_ans, re.DOTALL | re.IGNORECASE)
        if match_obj:
            final_ans.append(match_obj.group(1).strip())
        else:
            num_no_answer += 1
            final_ans.append("No answer")

    print(f"Number of no_answers: {num_no_answer}...")

    streamed_responses['final_ans'] = final_ans

    print('Saving final dataset...')
    main_dataset = main_dataset.merge(streamed_responses, on=['question'])
    main_dataset.to_json(base_path / "system2/final_response/final_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    parser.add_argument("--generation_mode", type=str, choices=['normal', 'ablation'], default='normal')
    args = parser.parse_args()
    main(dataset=args.dataset, generation_mode=args.generation_mode)