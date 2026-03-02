from argparse import ArgumentParser

from datasets import load_dataset, tqdm
from evaluate import load
import pandas as pd


def main(prediction_dataset_path: str, predicted_answer_field: str, ground_truth_field: str) -> None:
    squad_metric = load("squad_v2")
    predictions, references = [], []

    try:
        prediction_dataset = load_dataset('json', data_files=prediction_dataset_path, split='train')
        for idx, row in tqdm(enumerate(prediction_dataset)):
            pred = "" if row[predicted_answer_field] is None else row[predicted_answer_field]
            predictions.append({'prediction_text': pred, 'id': str(idx), 'no_answer_probability': 0.})
            references.append({'answers': {'answer_start': [0], 'text': [row[ground_truth_field]]}, 'id': str(idx)})
    except:
        prediction_dataset = pd.read_json(prediction_dataset_path, lines=True)
        for idx, row in tqdm(enumerate(prediction_dataset.itertuples())):
            pred = "" if getattr(row, predicted_answer_field) is None else getattr(row, predicted_answer_field)
            predictions.append({'prediction_text': pred, 'id': str(idx), 'no_answer_probability': 0.})
            references.append({'answers': {'answer_start': [0], 'text': [getattr(row, ground_truth_field)]}, 'id': str(idx)})

    results = squad_metric.compute(predictions=predictions, references=references)
    print(results)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_dataset_path", type=str)
    parser.add_argument("--predicted_answer_field", type=str)
    parser.add_argument("--ground_truth_field", type=str)
    args = parser.parse_args()
    main(prediction_dataset_path=args.prediction_dataset_path,
         predicted_answer_field=args.predicted_answer_field,
         ground_truth_field=args.ground_truth_field)