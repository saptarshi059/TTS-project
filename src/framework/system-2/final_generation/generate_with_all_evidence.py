from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os, torch
import sys
sys.path.append("../../../utils/")

from all_system_prompts import SYSTEM_2_MAIN_PROMPT
import re
import json


class System2Dataset(Dataset):
    def __init__(self, tokenizer, dataset, device):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.samples = []
        for row in tqdm(dataset.itertuples()):
            generated_triples_string = ", ".join(f"({triple})" for triple in row.generated_triples)
            retrieved_evidences = "\n\n".join(row.retrieved_docs)

            self.samples.append([{"role": "system", "content": SYSTEM_2_MAIN_PROMPT},
                                 {"role": "user", "content": f"<input>\n"
                                                             f"Question: {row.question}\n"
                                                             f"Initial (incorrect) Guess: {row.system_1_guess}\n"
                                                             f"Initial (incorrect) Reasoning (triples): {generated_triples_string}\n"
                                                             f"Counterfactual Evidence: {retrieved_evidences}\n"
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


def main(model_name:str, dataset:str, batch_size: int, gpu_id: str, output_dir: str) -> None:
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    main_dataset = pd.read_json(f"../../../../framework_output/system2/{dataset}/retrieval_results/retrieved_docs.jsonl", lines=True)
    op_dir = Path(output_dir) / f"{dataset}/final_response/"
    folder = Path(op_dir)
    folder.mkdir(parents=True, exist_ok=True)

    partial_op_file = Path(op_dir / "streamed_responses.jsonl")
    if partial_op_file.exists():
        completed_questions = pd.read_json(partial_op_file, lines=True)['question'].to_list()
        main_dataset = main_dataset.query("question not in @completed_questions")
        print(f"Resuming after completing {len(main_dataset)} questions...")

    print(f"Wrapping {dataset} with torch...")
    torch_dataset = System2Dataset(tokenizer=tokenizer, dataset=main_dataset, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    print(f"{'-'*10}Running System-2: MAIN GENERATION with {model_name} on {dataset}{'-'*10}")
    with Path(op_dir / "streamed_responses.jsonl").open("a") as file:
        for batch in tqdm(torch_dataset_dataloader):
            with torch.no_grad():
                generated_ids = model.generate(**batch, max_new_tokens=1024, do_sample=False, num_beams=1)
                decoded_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for generation in decoded_generation:
                    ques = re.findall(r"Question:(.*)", generation)[1].strip()
                    write_obj = {'question': ques, 'generation': generation}
                    json_string = json.dumps(write_obj)
                    file.write(json_string + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_directory", type=str, default="../../../../framework_output/system2/")
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size, gpu_id=args.gpu_id,
         output_dir=args.output_directory)