from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset, DataLoader
from datasets import tqdm, load_dataset
from argparse import ArgumentParser
from pathlib import Path
import os, torch
import sys
sys.path.append("../../utils/")

from all_system_prompts import SYSTEM_1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class System1Dataset(Dataset):
    def __init__(self, tokenizer, dataset, device):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

        self.samples = []
        for row in tqdm(dataset):
            self.samples.append([{"role": "system", "content": SYSTEM_1},
                                 {"role": "user", "content": f"QUESTION: {row['question']}"}])

        self.tokenized_samples = tokenizer.apply_chat_template(self.samples,
                                                               tokenize=False,
                                                               add_generation_prompt=True,
                                                               enable_thinking=False)

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

    base_path = Path("../../../data/")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    main_dataset = load_dataset("json", data_files=str(base_path / f"{dataset}/test.json"))["train"]
    torch_dataset = System1Dataset(tokenizer=tokenizer, dataset=main_dataset, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    raw_responses = []

    for batch in tqdm(torch_dataset_dataloader):
        with torch.no_grad():
            generated_ids = model.generate(**batch, max_new_tokens=20, do_sample=False)
            raw_responses.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    main_dataset = main_dataset.add_column("raw_responses", raw_responses)
    main_dataset.save_to_disk(base_path / f"{dataset}/raw_responses_from_system_1_phase")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="2wiki")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size)