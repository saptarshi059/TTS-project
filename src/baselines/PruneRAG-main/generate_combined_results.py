from argparse import ArgumentParser
import pandas as pd


def main(dataset):
    original_ds = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json")[['question', 'answer']]
    generated_ds = pd.read_json(f'outputs/{dataset}_outputs.jsonl', lines=True)
    generated_ds = generated_ds.merge(original_ds)

    generated_ds.to_json(f"outputs/{dataset}_parsed.jsonl", orient='records', lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(dataset=args.dataset)