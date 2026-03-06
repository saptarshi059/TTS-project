from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append("../../utils/")
from all_system_prompts import SYSTEM_1

def main(dataset: str):
    base_path = Path(f"../../../framework_output/system1/{dataset}")

    ds = pd.read_json(base_path / "raw_responses.jsonl", lines=True)

    print("Parsing raw responses...")
    cleaned_response = []
    for row in tqdm(ds.itertuples()):
        cleaned_response.append(row.raw_responses.split(SYSTEM_1)[-1].split("Answer:")[-1].strip())

    print("Saving cleaned responses...")
    ds['cleaned_ans'] = cleaned_response
    ds.to_json(base_path / "parsed_responses.jsonl", orient='records', lines=True, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)