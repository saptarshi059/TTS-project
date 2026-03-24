import pandas as pd
import argparse


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_dataset(self):
        print(f"Loading {self.dataset_name}")

        data_path = f"../../../sampled_data/{self.dataset_name}/sampled_ds.json"
        data = pd.read_json(data_path).to_dict('records')

        return data, data_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False)
    args = parser.parse_args()
    loader = DatasetLoader(args.dataset)
    loader.load_dataset()