from argparse import ArgumentParser
import pandas as pd


def main(dataset):
    original_ds = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json").loc[:, ['question', 'answer']]
    generated_ds = pd.read_json(f'outputs/{dataset}_outputs.jsonl', lines=True).loc[:, ['question', 'final_answer']]
    combined_df = pd.merge(original_ds, generated_ds)

    combined_df.to_json(f"outputs/{dataset}_parsed.jsonl", orient='records', lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(dataset=args.dataset)