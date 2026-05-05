from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import os, torch, sys, pandas as pd
sys.path.append("../../../utils/")

from all_system_prompts import TRIPLE_GEN

class TripleGenDataset(Dataset):
    def __init__(self, tokenizer, dataset, device):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.samples = []
        for row in tqdm(dataset.itertuples()):
            self.samples.append([{"role": "system", "content": TRIPLE_GEN},
                                 {"role": "user", "content": f"<input>\n"
                                                             f"Question: {row.question}\n"
                                                             f"Answer: {row.system_1_guess}\n"
                                                             f"</input>"}])
        self.tokenized_samples = tokenizer.apply_chat_template(self.samples, tokenize=False, add_generation_prompt=True)
        self.model_inputs = self.tokenizer(self.tokenized_samples, padding=True, return_tensors="pt")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.model_inputs["input_ids"][idx].to(self.device),
            "attention_mask": self.model_inputs["attention_mask"][idx].to(self.device),
        }


def main(model_name:str, dataset:str, batch_size: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    base_path = Path(f"../../../../framework_output/{dataset}")

    main_dataset = pd.read_json(base_path / "system1/system_2_start.jsonl", lines=True)
    print(f"Wrapping {dataset} with torch...")
    torch_dataset = TripleGenDataset(tokenizer=tokenizer, dataset=main_dataset, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    print(f"{'-'*10}Running TRIPLE GENERATION with {model_name} on {dataset}{'-'*10}")
    raw_responses = []
    for batch in tqdm(torch_dataset_dataloader):
        with torch.no_grad():
            generated_ids = model.generate(**batch, max_new_tokens=500, do_sample=False, num_beams=1)
            raw_responses.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    main_dataset["raw_responses"] = raw_responses

    print("Saving results...")
    op_dir = base_path / "system2/triple_extraction"
    op_dir.mkdir(parents=True, exist_ok=True)
    main_dataset.to_json(op_dir / "raw_responses.jsonl", lines=True, orient='records', index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size)