from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import sys
sys.path.append("../../../utils/")

from all_system_prompts import WHY, WHY_NOT


def clean_generation(base_df, question_col, gen_ds, strategy):
    all_user_prefix = {'why': "Please explain why this solution is correct:",
                       'why_not': "Please explain why this solution is incorrect:"
                       }
    user_prefix = all_user_prefix[strategy]

    all_system_prompt = {'why': WHY, 'why_not': WHY_NOT}
    system_prompt = all_system_prompt[strategy]

    stripped_gen = []
    for base_row, why_row in zip(base_df.itertuples(), gen_ds.itertuples()):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": rf"{user_prefix}\nQuestion: {base_row.get(question_col)}\nAnswer: {base_row.system_1_guess}"}]
        formatted_string = ""
        for element in messages:
            formatted_string += f"{element['role']}\n{element['content']}\n"
        formatted_string += 'assistant'

        stripped_gen.append(why_row.generation.split(formatted_string)[-1].strip())

    return stripped_gen

def main(dataset:str, question_column:str) -> None:
    dataset = dataset.replace("/", "_")
    base_dir = Path(f"../../../../all_output/{dataset}")

    base_ds = pd.read_json(Path(base_dir) / "system1_math/system_1_split_generations.jsonl", lines=True)
    why_ds = pd.read_json(Path(base_dir) / "why/streamed_responses.jsonl", lines=True)
    why_not_ds = pd.read_json(Path(base_dir) / "why_not/streamed_responses.jsonl", lines=True)

    why_gen_stripped = clean_generation(base_ds, question_column, why_ds, "why")
    why_not_gen_stripped = clean_generation(base_ds, question_column, why_not_ds, "why_not")

    base_ds['why_cleaned'] = why_gen_stripped
    base_ds['why_not_cleaned'] = why_not_gen_stripped

    op_dir = Path(base_dir) / "system2/"
    folder = Path(op_dir)
    folder.mkdir(parents=True, exist_ok=True)

    base_ds.to_json(base_dir / op_dir/"system2_start.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", choices=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--question_column", type=str, default="problem")
    args = parser.parse_args()
    main(dataset=args.dataset, question_column=args.question_column)