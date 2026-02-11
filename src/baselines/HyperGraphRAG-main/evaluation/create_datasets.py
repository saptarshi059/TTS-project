import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json


def return_gold_ctx(context, supporting_facts):
    gold = []
    for x in supporting_facts:
        ent, line = x[0], x[1]
        for y in context:
            if y[0] == ent:
                print(y)
                gold.append(y[1][line])
                break
    return gold


def main():
    base_path = Path("../../../../sampled_data")
    datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']

    for dataset_name in tqdm(datasets):
        print(f"Working on {dataset_name}...")
        dataset = pd.read_json(base_path / f"{dataset_name}/sampled_ds.json")

        print("Collecting elements...")
        all_ctx = []
        formatted_questions = []
        for row in dataset.itertuples():
            if dataset_name in {"2wikimultihopqa", "hotpotqa"}:
                context_list = return_gold_ctx(row.context, row.supporting_facts)
            else:
                context_dicts = list(filter(lambda x: x['is_supporting'], row.paragraphs))
                context_list = [x['paragraph_text'] for x in context_dicts]

            all_ctx.extend(context_list)
            formatted_questions.append({"question": row.question, "golden_answers": row.answer})

        print("Saving files...")
        Path("contexts").mkdir(parents=True, exist_ok=True)
        with open(f"contexts/{dataset_name}_contexts.json", 'w', encoding='utf-8') as f:
            json.dump(list(set(all_ctx)), f, indent=4, ensure_ascii=False)

        Path(f"datasets/{dataset_name}").mkdir(parents=True, exist_ok=True)
        with open(f"datasets/{dataset_name}/questions.json", 'w', encoding='utf-8') as f:
            json.dump(formatted_questions, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()