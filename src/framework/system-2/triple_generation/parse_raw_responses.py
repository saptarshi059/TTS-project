from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re

import sys
sys.path.append("../../../utils/")
from all_system_prompts import TRIPLE_GEN

def main(dataset: str):
    base_path = Path(f"../../../../framework_output/system2/{dataset}/triple_extraction")

    ds = pd.read_json(base_path / "raw_responses.jsonl", lines=True)

    print("Parsing raw responses...")
    generated_triples = []
    for row in tqdm(ds.itertuples()):
        ip_string = (f"<input>\n"
                     f"Question: {row.question}\n"
                     f"Explanation: {row.explanation}\n"
                     f"</input>")
        cleaned_string = row.raw_responses.split(TRIPLE_GEN)[1].strip().split(ip_string)[1].strip()
        generated_triples.append(re.findall(r"<triple>(.*?)</triple>", cleaned_string, re.IGNORECASE | re.DOTALL))

    print("Saving cleaned responses...")
    ds['generated_triples'] = generated_triples
    ds.to_json(base_path / "parsed_responses.jsonl", orient='records', lines=True, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='hotpotqa')
    args = parser.parse_args()
    main(dataset=args.dataset)