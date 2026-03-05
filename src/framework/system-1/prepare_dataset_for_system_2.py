import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

def main(dataset: str):
    base_path = Path(f"../../../framework_output/system1/{dataset}/system_1")

    ds = pd.read_json(base_path / "parsed_responses.jsonl", lines=True)
    completed = []
    for row in ds.itertuples():
        if row.answer == row.cleaned_ans:
            completed.append(row.question)
    print(f"Completed questions from system-1 thinking: {len(completed)} ({(len(completed)/len(ds) * 100):.2f}%)")

    print("Creating and saving dataset for system-2 processing...")
    filtered_ds = ds.query("question not in @completed")
    filtered_ds.to_json(base_path / "system_2_start.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    args = parser.parse_args()
    main(dataset=args.dataset)
