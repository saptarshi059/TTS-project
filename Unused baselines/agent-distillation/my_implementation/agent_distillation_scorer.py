from evaluate import load
from pathlib import Path
from datasets import load_from_disk, tqdm

def main():
    squad_metric = load("squad_v2")

    for folder in Path().glob("*_responses"):
        print(f"Working on {folder}...")
        response_dataset = load_from_disk(folder)
        predictions, references = [], []
        for idx, row in tqdm(enumerate(response_dataset)):
            predictions.append({'prediction_text': row['response'], 'id': str(idx), 'no_answer_probability': 0.})
            references.append({'answers': {'answer_start': [0], 'text': [row['gold_answers']]}, 'id': str(idx)})

        results = squad_metric.compute(predictions=predictions, references=references)
        print(f"EM: {results['exact']} | F1: {results['f1']:.2f}", f"\n{'-'*50}")


if __name__ == "__main__":
    main()