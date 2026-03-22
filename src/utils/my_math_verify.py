from math_verify import parse, verify
from argparse import ArgumentParser
from datasets import load_dataset
import pandas as pd
import statistics

def main(dataset:str, op_file: str):
    base_dataset = load_dataset(dataset, split='test').to_pandas()
    output_file = pd.read_json(op_file, lines=True)
    responses = []
    for base_row, response_row in zip(base_dataset.itertuples(), output_file.itertuples()):
        gold = parse(base_row.solution)
        answer = parse(response_row.generation)

        responses.append(verify(gold, answer))

    print(f"Accuracy: {statistics.mean(responses)*100:.2f}%")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str, default="../../../all_output/HuggingFaceH4_MATH-500/cot/formatted_generations.jsonl")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500")
    args = parser.parse_args()
    main(dataset=args.dataset, op_file=args.output_file)