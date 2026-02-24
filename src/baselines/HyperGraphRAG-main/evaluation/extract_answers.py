from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import re


def main(dataset):
    base_path = Path(f"results/HyperGraphRAG/{dataset}")
    df = pd.read_json(base_path / "test_result.json")

    final_answers = []
    for row in df.itertuples():
        s1 = re.search(r"<answer>(.*?)</answer>", row.generation, re.DOTALL)
        if s1 is not None:
            final_answers.append(s1.group(1).strip())
        else:
            try:
                s2 = re.search(r"<answer>(.*)", row.generation, re.DOTALL)
                final_answers.append(s2.group(1).split('<answer>')[0].strip())
            except: # There's just 1 edge case in hotpotqa that needs to be handled like this.
                final_answers.append(re.search(r"</think>(.*?)</answer>", row.generation, re.DOTALL).group(1).strip())

    df["final_answer"] = final_answers
    df.to_json(base_path / "test_parsed.jsonl", orient='records', lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(dataset=args.dataset)
