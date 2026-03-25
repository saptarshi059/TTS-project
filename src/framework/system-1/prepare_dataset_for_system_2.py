from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def main(dataset: str, confidence_threshold: int):
    base_path = Path(f"../../../framework_output/{dataset}/system1/")

    ds = pd.read_json(base_path / "parsed_responses.jsonl", lines=True)

    print(f"Evaluating {dataset}...")
    completed = []
    for row in tqdm(ds.itertuples()):
        if row.avg_log_prob >= confidence_threshold:
            completed.append(row.question)
    print(f"Completed questions from system-1 thinking: {len(completed)} ({(len(completed)/len(ds) * 100):.2f}%)")

    print("Saving datasets for system-1 complete and system-2 processing...")
    # System-1
    system_1_ds = ds.query("question in @completed")
    system_1_ds.to_json(base_path / "system_1_complete.jsonl", lines=True, orient='records', index=False)

    # System-2
    system_2_ds = ds.query("question not in @completed").copy()
    system_2_ds.to_json(base_path / "system_2_start.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(dataset=args.dataset, confidence_threshold=args.confidence_threshold)