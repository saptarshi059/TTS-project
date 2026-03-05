from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os, torch
import sys
sys.path.append("../../utils/")

from all_system_prompts import SYSTEM_1

class System1Dataset(Dataset):
    def __init__(self, tokenizer, dataset, device):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.samples = []
        for row in tqdm(dataset.itertuples()):
            self.samples.append([{"role": "system", "content": SYSTEM_1},
                                 {"role": "user", "content": f"QUESTION: {row.question}"}])
        self.tokenized_samples = tokenizer.apply_chat_template(self.samples, tokenize=False, add_generation_prompt=True)
        self.model_inputs = self.tokenizer(self.tokenized_samples, padding=True, return_tensors="pt")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.model_inputs["input_ids"][idx].to(self.device),
            "attention_mask": self.model_inputs["attention_mask"][idx].to(self.device),
        }


def main(model_name:str, dataset:str, batch_size: int, gpu_id: str, output_dir: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    main_dataset = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json")[['question', 'answer']]
    print(f"Wrapping {dataset} with torch...")
    torch_dataset = System1Dataset(tokenizer=tokenizer, dataset=main_dataset, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    print(f"{'-'*10}Running System-1 with {model_name} on {dataset}{'-'*10}")
    raw_responses = []
    for batch in tqdm(torch_dataset_dataloader):
        with torch.no_grad():
            generated_ids = model.generate(**batch, max_new_tokens=20)
            raw_responses.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    main_dataset["raw_responses"] = raw_responses

    print("Saving results...")
    op_dir = Path(output_dir) / f"{dataset}/system_1/"
    folder = Path(op_dir)
    folder.mkdir(parents=True, exist_ok=True)
    main_dataset.to_json(op_dir / "raw_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--output_directory", type=str, default="../../../framework_output/system1/")
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size, gpu_id=args.gpu_id,
         output_dir=args.output_directory)