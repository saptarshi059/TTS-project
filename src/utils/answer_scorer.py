from datasets import load_from_disk, tqdm
from argparse import ArgumentParser
from evaluate import load
import re

def main(prediction_dataset_path: str) -> None:
    def clean_prediction(text: str) -> str:
        match = re.search(r"<\/think>([\s\S]*)", text)
        answer = match.group(1) if match else None
        try:
            return answer.split('Answer:')[1].strip()
        except:
            return ""

    squad_metric = load("squad")
    prediction_dataset = load_from_disk(prediction_dataset_path)
    predictions, references = [], []
    for row in tqdm(prediction_dataset):
        predictions.append({'prediction_text': clean_prediction(row['raw_responses']), 'id': row['id']})
        references.append([{'answers': {'answer_start': [0], 'text': [row['answer']]}, 'id': row['id']}])

    results = squad_metric.compute(predictions=predictions, references=references)
    print(results)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction_dataset_path", type=str)
    args = parser.parse_args()
    main(prediction_dataset_path=args.prediction_dataset_path)