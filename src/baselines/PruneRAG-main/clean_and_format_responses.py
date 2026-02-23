from argparse import ArgumentParser
import pandas as pd
import re

def main(dataset_path):
    def clean_latex_text(text):
        return re.sub(r'\\+text\{', '', text).strip()

    df = pd.read_json(dataset_path, lines=True)
    df['extracted_answer'] = df['extracted_answer'].apply(clean_latex_text)

    final_answers = []
    for row in df.itertuples():
        if row.extracted_answer == '':
            final_answers.append(row.extracted_answer)
        else:
            final_answers.append(row.full_response)

    df['final_answer'] = final_answers
    df.to_json(dataset_path, orient='records', lines=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    args = parser.parse_args()
    main(dataset_path=args.response_file)