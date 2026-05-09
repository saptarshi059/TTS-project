import os
import sys
from argparse import ArgumentParser

import pandas as pd
import torch
from datasets import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.append("../../utils/")

from all_system_prompts import NAIVE_BASELINE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NaiveDataset(Dataset):
    def __init__(self, tokenizer, dataset, device):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

        self.samples = []
        for row in tqdm(dataset.itertuples()):
            supporting_facts = "\n".join(doc for doc in row['retrieved_docs'])
            self.samples.append([{"role": "system", "content": NAIVE_BASELINE},
                                 {"role": "user", "content": f"RELATED CONTEXT:\n{supporting_facts}\n\nQUESTION: {row['question']}"}])

        self.tokenized_samples = tokenizer.apply_chat_template(
            self.samples,
            tokenize=False,
            add_generation_prompt=True,
        )

        self.model_inputs = self.tokenizer(self.tokenized_samples, padding=True, return_tensors="pt")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.model_inputs["input_ids"][idx].to(self.device),
            "attention_mask": self.model_inputs["attention_mask"][idx].to(self.device),
        }


def main(model_name:str, dataset:str, batch_size: int) -> None:
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    main_dataset = pd.read_json(f"{dataset}_with_retrieved_triples_from_naive_baseline.json")
    torch_dataset = NaiveDataset(tokenizer=tokenizer, dataset=main_dataset, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    raw_responses = []

    for batch in tqdm(torch_dataset_dataloader):
        with torch.no_grad():
            generated_ids = model.generate(**batch, max_new_tokens=1000, do_sample=False)
            raw_responses.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    main_dataset["raw_responses"] = raw_responses
    main_dataset.to_json(f"{dataset}_with_raw_responses_from_naive_baseline.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="2wikimultihopqa")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size)