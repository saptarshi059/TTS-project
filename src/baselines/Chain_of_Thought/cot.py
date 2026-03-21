from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os, torch
import sys
sys.path.append("../../utils/")

from all_system_prompts import COT

class CoTDataset(Dataset):
    def __init__(self, tokenizer, dataset, device):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.samples = []
        for row in tqdm(dataset.itertuples()):
            self.samples.append([{"role": "system", "content": COT},
                                 {"role": "user", "content": rf"{row.problem} \n\nPlease put your final numerical or algebraic answer inside \boxed{{}}."}])
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
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    ds = load_dataset(dataset, split='test').to_pandas()
    dataset = dataset.replace("/", "_")
    print(f"Wrapping {dataset} with torch...")
    torch_dataset = CoTDataset(tokenizer=tokenizer, dataset=ds, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    print(f"{'-'*10}Running CoT with {model_name} on {dataset}{'-'*10}")
    raw_responses = []
    for batch in tqdm(torch_dataset_dataloader):
        with torch.no_grad():
            generated_ids = model.generate(**batch, max_new_tokens=50, do_sample=False, num_beams=1)
            raw_responses.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    ds["raw_responses"] = raw_responses

    print("Saving results...")
    op_dir = Path(output_dir) / f"{dataset}"
    folder = Path(op_dir)
    folder.mkdir(parents=True, exist_ok=True)
    ds.to_json(op_dir / "raw_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", choices=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_directory", type=str, default="../../../all_output/cot/")
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size, gpu_id=args.gpu_id,
         output_dir=args.output_directory)