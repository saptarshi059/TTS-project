from datasets import load_dataset
import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    datasets = {#'2wikimultihopqa': 'dev.json',
                #'hotpotqa': 'hotpot_dev_fullwiki_v1.json',
                #'musique': 'musique_ans_v1.0_dev.jsonl',
                #'frames': 'google/frames-benchmark',
                'triviaqa': 'hf://datasets/mandarjoshi/trivia_qa/rc.wikipedia/validation-00000-of-00001.parquet'}

    for dataset_name, dataset_file_name in datasets.items():
        print(f'Processing {dataset_name}...')

        if dataset_name in {'2wikimultihopqa', 'hotpotqa'}:
            dataset = pd.read_json(f"../../all_data/{dataset_name}/{dataset_file_name}")
            column_name = 'question'
        elif dataset_name == 'musique':
            dataset = pd.read_json(f"../../all_data/{dataset_name}/{dataset_file_name}", lines=True)
            column_name = 'question'
        elif dataset_name == 'frames':
            dataset = load_dataset(dataset_file_name, split='test').to_pandas()
            column_name = 'Prompt'
        elif dataset_name == 'triviaqa':
            with pq.ParquetFile(dataset_file_name) as pf:
                first_group = pf.read_row_group(0)
                dataset = first_group.to_pandas()
            column_name = 'question'

        dataset.drop_duplicates(subset=column_name, inplace=True)
        if dataset_name in {'2wikimultihopqa', 'hotpotqa'}:
            sampled_ds = dataset.sample(n=1000, random_state=42)
            final_rows = []
            for row in sampled_ds.itertuples():
                if len(final_rows) == 500:
                    break

                try:
                    # Have to do this because some samples have wrong gold sentence indices.
                    gold = []
                    for x in row.supporting_facts:
                        ent, line = x[0], x[1]
                        for y in row.context:
                            if y[0] == ent:
                                gold.append(y[1][line])
                                break
                    final_rows.append(row)
                except:
                    continue

            sampled_ds = pd.DataFrame(final_rows)
        else:
            sampled_ds = dataset.sample(n=500, random_state=42)

        sampled_ds.rename(columns={'Prompt':'question',
                                   'Answer': 'answer',
                                   'Index': 'id',
                                   'Unnamed: 0': 'id'}, inplace=True)

        if dataset_name == 'triviaqa':
            flattened_answer = []
            for row in sampled_ds.itertuples():
                current_row_ans = []
                for key, value in row.answer.items():
                    if key == 'type':
                        continue

                    if type(value) is np.ndarray:
                        current_row_ans.extend(value.tolist())
                    else:
                        current_row_ans.append(value)

                current_row_ans = list(set(current_row_ans))
                current_row_ans = list(filter(lambda s: s != "", current_row_ans))

                flattened_answer.append(current_row_ans)

            sampled_ds.drop(columns=['answer'], inplace=True)
            sampled_ds['answer'] = flattened_answer

        if 'id' in sampled_ds.columns:
            sampled_ds = sampled_ds.astype({'id': str})
        op_dir = Path(f"../../sampled_data/{dataset_name}")
        op_dir.mkdir(parents=True, exist_ok=True)
        sampled_ds.to_json(op_dir / "sampled_ds.json", index=False)


if __name__ == "__main__":
    main()