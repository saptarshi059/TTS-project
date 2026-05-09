import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.append("../../../utils/")

from all_system_prompts import SYSTEM_2
import json


class System2Dataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        question = sample.question
        gold_answer = sample.gold_answer
        s1_guess = sample['system_1_guess']

        generated_triples_string = ", ".join(f"({triple})" for triple in sample.generated_triples)
        retrieved_evidences = "\n\n".join(dict.fromkeys(sample.retrieved_docs))  # dedup, stable order

        messages = [{"role": "system", "content": SYSTEM_2},
                    {"role": "user", "content": f"<input>\n"
                                                f"Question: {question}\n"
                                                f"Initial Guess: {sample.system_1_guess}\n"
                                                f"Initial Reasoning: {generated_triples_string}\n"
                                                f"Retrieved Context: {retrieved_evidences}\n"
                                                f"</input>"}]
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return {"text": formatted_text, "question": question, "gold_answer": gold_answer, "system_1_guess": s1_guess}


def custom_collate_fn(batch, tokenizer, device):
    texts = [item["text"] for item in batch]
    questions = [item["question"] for item in batch]
    gold_answers = [item["gold_answer"] for item in batch]
    s1_guesses = [item["system_1_guess"] for item in batch]

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    return {
        "question": questions,
        "gold_answer": gold_answers,
        "s1_guesses": s1_guesses,
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"]
    }


def main(model_name:str, dataset:str, batch_size: int) -> None:
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    base_path = Path(f"../../../../framework_output/{dataset}/system2")

    ds = pd.read_json(base_path / "retrieval_results/with_retrieved_docs.jsonl", lines=True)

    op_dir = base_path / "final_response"
    op_dir.mkdir(parents=True, exist_ok=True)

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
    torch_dataset = System2Dataset(tokenizer=tokenizer, dataset=ds)

    # Create a version of the function that already knows the tokenizer
    collate_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_with_tokenizer)


    print(f"{'-'*10}Running System-2: MAIN GENERATION with {model_name} on {dataset}{'-'*10}")
    with Path(op_dir / "streamed_responses.jsonl").open("a") as file:
        for batch in tqdm(torch_dataset_dataloader):
            batch_questions = batch.pop('question')
            batch_gold_ans = batch.pop('gold_answer')
            batch_s1_guess = batch.pop('s1_guesses')
            with torch.no_grad():
                generated_ids = model.generate(**batch, max_new_tokens=1000, do_sample=False, num_beams=1)
                decoded_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for ques, gold, s1_g, generation in zip(batch_questions, batch_gold_ans, batch_s1_guess, decoded_generation):
                    write_obj = {'question': ques,
                                 'gold_answer':gold,
                                 'system_1_guess': s1_g,
                                 'generation': generation}
                    json_string = json.dumps(write_obj)
                    file.write(json_string + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size)