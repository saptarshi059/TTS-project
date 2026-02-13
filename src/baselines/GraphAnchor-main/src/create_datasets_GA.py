from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json


def main():
    datasets = ['2wikimultihopqa', 'hotpotqa', 'musique', 'frames']
    base_path = Path("../../../../sampled_data")

    for dataset_name in tqdm(datasets):
        print(f"Working on {dataset_name}...")
        dataset = pd.read_json(base_path / f"{dataset_name}/sampled_ds.json")

        formatted_samples = []
        for row in dataset.itertuples():
            if isinstance(row.answer, str):
                answer_list = [row.answer]
            else:
                answer_list = row.answer

            formatted_samples.append({"question": row.question, "answer": answer_list})

        Path(f"../data/eval/{dataset_name}").mkdir(parents=True, exist_ok=True)
        with open(f"../data/eval/{dataset_name}/test.json", 'w', encoding='utf-8') as f:
            json.dump(formatted_samples, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()