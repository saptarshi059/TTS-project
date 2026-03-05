from argparse import ArgumentParser
from evaluate import load
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def main(dataset: str):
    squad_metric = load("squad_v2")
    base_path = Path(f"../../../framework_output/system1/{dataset}/system_1")

    ds = pd.read_json(base_path / "parsed_responses.jsonl", lines=True)
    completed = []
    for idx, row in tqdm(enumerate(ds.itertuples())):
        prediction = [{'prediction_text': row.cleaned_ans, 'id': str(idx), 'no_answer_probability': 0.}]
        reference = [{'answers': {'answer_start': [0], 'text': [row.answer]}, 'id': str(idx)}]

        score = squad_metric.compute(predictions=prediction, references=reference)
        if score['exact'] == 100.0:
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
