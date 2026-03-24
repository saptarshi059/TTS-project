from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re

def main(dataset: str):
    base_path = Path(f"../../../../framework_output/system2/{dataset}/explanation_extraction")

    ds = pd.read_json(base_path / "raw_responses.jsonl", lines=True)

    print("Parsing raw responses...")
    explanation = []
    for row in tqdm(ds.itertuples()):
        explanation.append(re.findall(r"<output>(.*?)</output>", row.raw_responses, re.IGNORECASE | re.DOTALL)[-1].strip())

    print("Saving cleaned responses...")
    ds['explanation'] = explanation
    ds.to_json(base_path / "parsed_responses.jsonl", orient='records', lines=True, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='2wikimultihopqa')
    args = parser.parse_args()
    main(dataset=args.dataset)