from argparse import ArgumentParser
import pandas as pd
import re

def main(dataset_path):
    df = pd.read_json(dataset_path, lines=True)

    def clean_latex_text(text):
        return re.sub(r'\\+text\{', '', text).strip()

    df['extracted_answer'] = df['extracted_answer'].apply(clean_latex_text)

    df.to_json(dataset_path, orient='records', lines=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    args = parser.parse_args()
    main(dataset_path=args.response_file)