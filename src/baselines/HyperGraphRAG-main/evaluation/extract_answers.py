from argparse import ArgumentParser
import pandas as pd
import re


def main(dataset_path):
    df = pd.read_json(dataset_path)

    final_answers = []
    for row in df.itertuples():
        s1 = re.search(r"<answer>(.*?)</answer>", row.generation, re.DOTALL)
        if s1 is not None:
            final_answers.append(s1.group(1).strip())
        else:
            # try s2
            s2 = re.search(r"<answer>(.*)", row.generation, re.DOTALL)
            final_answers.append(s2.group(1).split('<answer>')[0].strip())

    df["final_answer"] = final_answers
    df.to_json(dataset_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    args = parser.parse_args()
    main(dataset_path=args.dataset_path)
