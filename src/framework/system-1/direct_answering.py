from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from datasets import load_dataset
from functools import partial
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os, torch
import json
import sys
sys.path.append("../../utils/")

from all_system_prompts import *

class GenerationDataset(Dataset):
    def __init__(self, tokenizer, dataset, system_prompt, dataset_type, question_column):
        self.tokenizer = tokenizer
        self.questions = dataset.get(question_column).to_list()
        self.system_prompt = system_prompt
        self.all_user_suffix = {'math': r"\n\nPlease put your final numerical or algebraic answer inside \boxed{}.",
                                'commonsense': ""
                                }
        self.user_suffix = self.all_user_suffix[dataset_type]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Question: {question} {self.user_suffix}"}]
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return {
            "text": formatted_text,
            "question": question
        }


def custom_collate_fn(batch, tokenizer, device):
    texts = [item["text"] for item in batch]
    questions = [item["question"] for item in batch]

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    return {
        "question": questions,
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"]
    }


def main(model_name:str, strategy:str, dataset:str, question_column: str, batch_size: int,
         gpu_id: str, output_dir: str) -> None:
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    all_system_prompts = {'cot': COT, 'system1': SYSTEM_1_MATH}
    system_prompt = all_system_prompts[strategy]

    all_dataset_type = {"HuggingFaceH4/MATH-500": "math"}
    dataset_type = all_dataset_type[dataset]

    all_max_tokens = {'cot': 1000, 'system1': 20}
    max_tokens = all_max_tokens[strategy]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    ds = load_dataset(dataset, split='test').to_pandas()
    dataset = dataset.replace("/", "_")

    op_dir = Path(output_dir) / f"{dataset}/{strategy}/"
    folder = Path(op_dir)
    folder.mkdir(parents=True, exist_ok=True)

    partial_op_file = op_dir / "streamed_responses.jsonl"
    if partial_op_file.exists():
        try:
            completed_df = pd.read_json(partial_op_file, lines=True)

            if not completed_df.empty:
                completed_questions = set(completed_df['question'].tolist())
                print(f"Completed Questions: {len(completed_questions)}...")

                ds = ds[~ds['question'].isin(completed_questions)]
                print(f"Questions remaining: {len(ds)}...")
            else:
                print("Completed file is empty. Proceeding with all questions.")

        except Exception as e:
            print(f"Error reading partial file: {e}. Starting from scratch.")
    else:
        print("No existing progress found. Starting fresh.")

    print(f"Wrapping {dataset} with torch...")
    torch_dataset = GenerationDataset(tokenizer=tokenizer, dataset=ds, system_prompt=system_prompt,
                                      dataset_type=dataset_type, question_column=question_column)

    print(f"{'-'*10}FORMATTED DATASET SAMPLE: {torch_dataset[0]['text']} {'-'*10}")

    # Create a version of the function that already knows the tokenizer
    collate_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_with_tokenizer)

    print(f"{'-'*10}Running {strategy} with {model_name} on {dataset}{'-'*10}")
    with Path(op_dir / "streamed_responses.jsonl").open("a") as file:
        for batch in tqdm(torch_dataset_dataloader):
            batch_questions = batch.pop('question')
            with torch.no_grad():
                generated_ids = model.generate(**batch, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
                decoded_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for ques, generation in zip(batch_questions, decoded_generation):
                write_obj = {'question': ques, 'generation': generation}
                json_string = json.dumps(write_obj)
                file.write(json_string + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--strategy", type=str, default='system1', choices=['system1', 'cot'])
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", choices=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--question_column", type=str, default="problem")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_directory", type=str, default="../../../all_output/")
    args = parser.parse_args()
    main(model_name=args.model_name, strategy=args.strategy, question_column=args.question_column,
         dataset=args.dataset, batch_size=args.batch_size, gpu_id=args.gpu_id,
         output_dir=args.output_directory)