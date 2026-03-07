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
    base_path = Path(f"../../../framework_output/system1/{dataset}")

    ds = pd.read_json(base_path / "raw_responses.jsonl", lines=True)

    print("Parsing raw responses...")
    cleaned_response = []
    for row in tqdm(ds.itertuples()):
        cleaned_string = row.raw_responses.split(SYSTEM_1)[-1].strip()
        match = re.search(r'<answer>(.*)</answer>', cleaned_string , re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_response.append(match.group(1).strip())
        else:
            # Trying as best as possible to get the main output
            ip_string = f"<input>\nQuestion: {row.question}</input>"
            cleaned_string = cleaned_string.split(ip_string)[1].strip()
            cleaned_string = re.sub(r'<\|.*?\|>|assistant\n', '', cleaned_string)
            soup = BeautifulSoup(cleaned_string, "html.parser")
            clean_text = soup.get_text(strip=True)
            cleaned_response.append(clean_text)

    print("Saving cleaned responses...")
    ds['cleaned_ans'] = cleaned_response
    ds.to_json(base_path / "parsed_responses.jsonl", orient='records', lines=True, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)