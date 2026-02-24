from argparse import ArgumentParser
from datasets import Dataset
import pandas as pd


def main(dataset):
    original_ds = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json")
    generated_ds = pd.read_json(f'outputs/{dataset}_outputs.jsonl', lines=True)
    combined_df = pd.merge(original_ds, generated_ds)
    combined_df = Dataset.from_pandas(combined_df)
    combined_df.to_json(f"outputs/{dataset}_parsed.jsonl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(dataset=args.dataset)