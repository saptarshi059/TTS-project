import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import ast

def flatten_deep(items):
    """Yield items from any nested iterable."""
    for x in items:
        # Check if an item is an iterable but not a string or bytes
        if isinstance(x, list):
            yield from flatten_deep(x)
        else:
            yield x


def main():
    base_path = Path("../../../../sampled_data")
    datasets = ['2wikimultihopqa', 'frames', 'hotpotqa', 'musique']

    frames_corpus = pd.read_parquet("frames_corpus")

    for dataset_name in tqdm(datasets):
        print(f"Working on {dataset_name}...")
        dataset = pd.read_json(base_path / f"{dataset_name}/sampled_ds.json")

        print("Collecting elements...")
        all_ctx = []
        formatted_questions = []
        for row in dataset.itertuples():
            if dataset_name == 'frames':
                row_links = ast.literal_eval(row.wiki_links)
                context_list = list(frames_corpus[frames_corpus["URL"].isin(row_links)].Text)
            elif dataset_name in {"2wikimultihopqa", "hotpotqa"}:
                context_list = row.context
            else:
                context_dicts = list(filter(lambda x: x['is_supporting'], row.paragraphs))
                context_list = [x['paragraph_text'] for x in context_dicts]

            all_ctx.extend(flatten_deep(context_list))
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