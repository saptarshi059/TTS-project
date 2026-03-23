from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import sys
sys.path.append("../../../utils/")

from all_system_prompts import WHY, WHY_NOT

sample = self.dataset.iloc[idx]
question = sample.get(self.question_column)
answer = sample.get("system_1_guess")

messages = [{"role": "system", "content": WHY},
            {"role": "user", "content": rf"{self.user_prefix}\nQuestion: {question}\nAnswer: {answer}"}]
formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main(strategy:str, dataset:str, question_column: str) -> None:

    all_user_prefix = {'why': "Please explain why this solution is correct:",
                       'why_not': "Please explain why this solution is incorrect:"
                       }

    dataset = dataset.replace("/", "_")
    base_dir = ""
    ds = pd.read_json(Path(output_dir) / f"{dataset}/system1_math/system_1_split_generations.jsonl", lines=True)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", choices=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--question_column", type=str, default="problem")
    args = parser.parse_args()
    main(strategy=args.strategy, question_column=args.question_column, dataset=args.dataset)