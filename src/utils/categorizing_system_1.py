from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

import torch, pandas as pd

from all_system_prompts import CATEGORIZATION_PROMPT

class GenerationDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        question = sample['question']
        gold_answer = sample['answer']
        system_1_guess = sample['system_1_guess']
        messages = [{"role": "system", "content": CATEGORIZATION_PROMPT},
                    {"role": "user", "content": f"<input>\n"
                                                f"Question: {question}\n"
                                                f"Gold Answer: {gold_answer}\n"
                                                f"Predicted Answer: {system_1_guess}\n"
                                                f"</input>"}]

        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(formatted_text, padding=True, return_tensors="pt")

        return model_inputs


if __name__ == "__main__":
    base_path = Path("../../experiment_runs/main_framework_run")

    w = pd.read_json(base_path / "2wikimultihopqa/system1/parsed_responses.jsonl", lines=True)
    h = pd.read_json(base_path / "hotpotqa/system1/parsed_responses.jsonl", lines=True)
    m = pd.read_json(base_path / "musique/system1/parsed_responses.jsonl", lines=True)
    combined_outputs_df = pd.concat([w, h, m])

    set_seed(42)
    judge = "Qwen/Qwen3-32B"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=judge, padding_side='left')
    '''model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=judge,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")'''

    print(f"Wrapping predictions dataframe with torch...")
    torch_dataset = GenerationDataset(tokenizer=tokenizer, dataset=combined_outputs_df)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=8, shuffle=False)

