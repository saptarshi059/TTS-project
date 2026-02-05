import pandas as pd
from datasets import load_dataset

def main():
    datasets = {'2wikimultihopqa': 'dev.json',
                'hotpotqa': 'hotpot_dev_fullwiki_v1.json',
                'musique': 'musique_ans_v1.0_dev.jsonl',
                'frames': 'google/frames-benchmark'}

    for dataset_name, dataset_file_name in datasets.items():
        if dataset_name in {'2wikimultihopqa', 'hotpotqa'}:
            dataset = pd.read_json(f"../../all_data/{dataset_name}/{dataset_file_name}")
            column_name = 'question'
        elif dataset_name == 'musique':
            dataset = pd.read_json(f"../../all_data/{dataset_name}/{dataset_file_name}", lines=True)
            column_name = 'question'
        else:
            dataset = load_dataset(dataset_file_name, split='test').to_pandas()
            column_name = 'Prompt'

        dataset.drop_duplicates(subset=column_name, inplace=True)
        sampled_ds = dataset.sample(n=500, random_state=42)
        sampled_ds.to_json(f"../../sampled_data/{dataset_name}/sampled_ds.json")

if __name__ == "__main__":
    main()