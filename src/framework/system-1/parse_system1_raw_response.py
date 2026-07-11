from argparse import ArgumentParser
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re

import sys
sys.path.append("../../utils/")
from all_system_prompts import RANDOM_PROMPT

def main(dataset: str):
    generation_folder_path = Path(f"../../../framework_output/{dataset}/system1/")
    generation_ds = pd.read_json(generation_folder_path / "streamed_responses.jsonl", lines=True)

    print("Parsing raw responses...")
    cleaned_response = []
    for row in tqdm(generation_ds.itertuples()):
        cleaned_string = row.generation.split(RANDOM_PROMPT)[-1].strip()
        match = re.search(r'<answer>(.*)</answer>', cleaned_string , re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_response.append(match.group(1).strip())
        else:
            # Trying as best as possible to get the main output
            ip_string = f"<input>\nQuestion: {row.question}\n</input>"
            cleaned_string = cleaned_string.split(ip_string)[1].strip()
            cleaned_string = re.sub(r'<\|.*?\|>|assistant\n', '', cleaned_string)
            soup = BeautifulSoup(cleaned_string, "html.parser")
            clean_text = soup.get_text(strip=True)
            cleaned_response.append(clean_text)

    print("Saving cleaned responses...")
    generation_ds['system_1_guess'] = cleaned_response
    generation_ds = generation_ds[['question', 'gold_answer', 'system_1_guess', 'avg_log_prob']]
    generation_ds.to_json(generation_folder_path / "parsed_responses.jsonl", orient='records', lines=True, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)