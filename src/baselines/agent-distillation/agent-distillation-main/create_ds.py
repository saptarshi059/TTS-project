import pandas as pd
import json
import os

for dataset_name in ['2wikimultihopqa', 'musique', 'frames', 'hotpotqa', 'triviaqa']:
    formatted_samples = []
    ds = pd.read_json(f"../../../../sampled_data/{dataset_name}/sampled_ds.json")
    for idx, row in enumerate(ds.itertuples()):
        formatted_samples.append({
            "id": row.id if 'id' in ds.columns else str(idx),
            "question": row.question,
            "answer": row.answer,
            "level": "hard",
            "type": "compositional",
            "dataset_name": dataset_name,
            "split": "test"
        })

    data = {"metadata": {
        "dataset_info": {
            "name": dataset_name,
            "fold": "test",
            "examples_range": {
                "start": 0,
                "end": 1000
            },
            "total_examples": 500,
            "creation_date": "2025-05-11",
            "filtering_criteria": None
        }
    },
    "examples": formatted_samples}

    folder_path = "data_processor/qa_dataset/test/"
    os.makedirs(folder_path, exist_ok=True)

    with open(f'{folder_path}/{dataset_name}.json', 'w') as f:
        json.dump(data, f, indent=4)