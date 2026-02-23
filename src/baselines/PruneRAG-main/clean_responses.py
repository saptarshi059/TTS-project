from argparse import ArgumentParser
import pandas as pd

def main(dataset_path):
    ds = pd.read_json(dataset_path, lines=True)
    for row in ds.itertuples():
        if '\text{' in row.extracted_answer:
            row.extracted_answer = row.extracted_answer.rstrip('\text{')
        elif '\\text{' in row.extracted_answer:
            row.extracted_answer = row.extracted_answer.rstrip('\\text{')

    ds.to_json(dataset_path, index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--response_file", type=str)
    args = parser.parse_args()
    main(dataset_path=args.response_file)