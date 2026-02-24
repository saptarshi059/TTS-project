from argparse import ArgumentParser
import pandas as pd


def main(dataset):
    original_ds = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json")
    generated_ds = pd.read_json(f'outputs/{dataset}_outputs.jsonl', lines=True)
    combined_df = pd.merge(original_ds, generated_ds).drop(columns=['id', '_1', 'type', 'context', 'entity_ids',
       'supporting_facts', 'evidences', 'evidences_id', 'answer_id', 'request_id'])

    combined_df.to_json(f"outputs/{dataset}_parsed.jsonl", orient='records', lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(dataset=args.dataset)