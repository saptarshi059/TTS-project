from argparse import ArgumentParser
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re

import sys
sys.path.append("../../utils/")
from all_system_prompts import SYSTEM_1

def main(dataset: str):
    base_ds = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json")[['question', 'answer']]

    generation_folder_path = Path(f"../../../framework_output/{dataset}/system1/")
    generation_ds = pd.read_json(generation_folder_path / "streamed_responses.jsonl", lines=True)

    merged_df = pd.merge(base_ds, generation_ds)

    print("Parsing raw responses...")
    cleaned_response = []
    for row in tqdm(merged_df.itertuples()):
        cleaned_string = row.generation.split(SYSTEM_1)[-1].strip()
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
    merged_df['system_1_guess'] = cleaned_response
    merged_df.to_json(generation_folder_path / "parsed_responses.jsonl", orient='records', lines=True, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)