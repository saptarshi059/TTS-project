import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from functools import partial

import pandas as pd
import torch
from datasets import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.append("../../utils/")

from all_system_prompts import NAIVE_BASELINE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NaiveDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        question = sample.question
        supporting_facts = "\n".join(doc for doc in sample['retrieved_docs'])
        messages = [{"role": "system", "content": NAIVE_BASELINE},
                    {"role": "user", "content": f"RELATED CONTEXT:\n{supporting_facts}\nQUESTION: {question}"}]
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"text": formatted_text, "question": question}


def custom_collate_fn(batch, tokenizer, device):
    texts = [item["text"] for item in batch]
    questions = [item["question"] for item in batch]

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    return {
        "question": questions,
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"]
    }


def main(model_name:str, dataset:str, batch_size: int) -> None:
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    ds = pd.read_json(f"{dataset}_with_retrieved_triples_from_naive_baseline.json")

    partial_op_file = Path(f"{dataset}_streamed_responses.jsonl")
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

    torch_dataset = NaiveDataset(tokenizer=tokenizer, dataset=ds)

    # Create a version of the function that already knows the tokenizer
    collate_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_with_tokenizer)

    with Path(f"{dataset}_streamed_responses.jsonl").open("a") as file:
        for batch in tqdm(torch_dataset_dataloader):
            batch_questions = batch.pop('question')
            with torch.no_grad():
                generated_ids = model.generate(**batch, max_new_tokens=1000, do_sample=False)
                decoded_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for ques, generation in zip(batch_questions, decoded_generation):
                    write_obj = {'question': ques, 'generation': generation}
                    json_string = json.dumps(write_obj)
                    file.write(json_string + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="2wikimultihopqa")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset_name, batch_size=args.batch_size)