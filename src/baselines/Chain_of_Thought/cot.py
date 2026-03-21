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

from all_system_prompts import COT

class CoTDataset(Dataset):
    def __init__(self, tokenizer, dataset, question_column):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.questions = dataset.get(question_column).to_list()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        formatted_sample = [{"role": "system", "content": COT},
                            {"role": "user", "content": rf"Question: {question} \n\nPlease put your final numerical or algebraic answer inside \boxed{{}}."}]

        tokenized_sample = self.tokenizer.apply_chat_template(formatted_sample, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(tokenized_sample, return_tensors="pt")

        return {
            "question": question,
            "input_ids": model_inputs["input_ids"].squeeze(0), # Remove batch dim; DataLoader will re-batch
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
        }


def custom_collate_fn(batch, tokenizer):
    # 1. Separate the text questions from the tensors
    questions = [item["question"] for item in batch]

    # 2. Grab the input_ids and attention_masks
    # We need to remove the extra '1' dimension added by return_tensors="pt" in __getitem__
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    attention_mask = [item["attention_mask"].squeeze(0) for item in batch]

    # 3. Use the tokenizer to pad the tensors
    # This creates the actual batch tensor
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt"
    )

    return {
        "question": questions,  # Returns as a list of strings
        "input_ids": padded_inputs["input_ids"],  # Returns as a Padded Tensor
        "attention_mask": padded_inputs["attention_mask"]  # Returns as a Padded Tensor
    }


def main(model_name:str, dataset:str, question_column: str, batch_size: int, gpu_id: str, output_dir: str) -> None:
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    ds = load_dataset(dataset, split='test').to_pandas()
    dataset = dataset.replace("/", "_")

    op_dir = Path(output_dir) / f"{dataset}/cot/"
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
    torch_dataset = CoTDataset(tokenizer=tokenizer, dataset=ds, question_column=question_column)

    # Create a version of the function that already knows the tokenizer
    collate_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_with_tokenizer)

    print(f"{'-'*10}Running CoT with {model_name} on {dataset}{'-'*10}")
    with Path(op_dir / "streamed_responses.jsonl").open("a") as file:
        for batch in tqdm(torch_dataset_dataloader):
            batch_questions = batch.pop('question')
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                generated_ids = model.generate(**batch, max_new_tokens=1000, do_sample=False, num_beams=1)
                decoded_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for ques, generation in zip(batch_questions, decoded_generation):
                write_obj = {'question': ques, 'generation': generation}
                json_string = json.dumps(write_obj)
                file.write(json_string + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500", choices=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--question_column", type=str, default="problem")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--output_directory", type=str, default="../../../all_output/")
    args = parser.parse_args()
    main(model_name=args.model_name, question_column=args.question_column,
         dataset=args.dataset, batch_size=args.batch_size, gpu_id=args.gpu_id,
         output_dir=args.output_directory)