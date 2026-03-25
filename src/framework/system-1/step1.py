from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os, torch
import json
import sys
sys.path.append("../../utils/")

from all_system_prompts import SYSTEM_1

class GenerationDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.questions = dataset['question'].to_list()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        messages = [{"role": "system", "content": SYSTEM_1},
                    {"role": "user", "content": f"<input>\n"
                                                f"Question: {question}\n"
                                                f"</input>"}]
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


def main(model_name:str, dataset:str, batch_size: int, gpu_id: str) -> None:
    set_seed(42)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    ds = pd.read_json(f"../../../sampled_data/{dataset}/sampled_ds.json")[['question', 'answer']]

    op_dir = Path(f"../../../framework_output/{dataset}/system1/")
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
    torch_dataset = GenerationDataset(tokenizer=tokenizer, dataset=ds)

    print(f"{'-'*10}FORMATTED DATASET SAMPLE{'-'*10}\n{torch_dataset[0]['text']}\n{'-'*10}")

    # Create a version of the function that already knows the tokenizer
    collate_with_tokenizer = partial(custom_collate_fn, tokenizer=tokenizer, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_with_tokenizer)

    print(f"{'-'*10}Running SYSTEM-1 with {model_name} on {dataset}{'-'*10}")
    with Path(op_dir / "streamed_responses.jsonl").open("a") as file:
        for batch in tqdm(torch_dataset_dataloader):
            batch_questions = batch.pop('question')

            with torch.no_grad():
                # 1. Ask for scores and the full output dict
                outputs = model.generate(
                    **batch,
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                generated_ids = outputs.sequences
                # 2. Extract scores (logits). This is a tuple of length = max_new_tokens
                # Each element is a tensor of shape (batch_size, vocab_size)
                logits = torch.stack(outputs.scores, dim=1)

                # 3. Calculate confidence (e.g., Mean Log Probability)
                # We use log_softmax to get normalized probabilities
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Gather the log-probs of the actual tokens generated
                # We shift generated_ids because scores start from the first generated token
                token_log_probs = torch.gather(
                    log_probs,
                    2,
                    generated_ids[:, -logits.shape[1]:].unsqueeze(-1)
                ).squeeze(-1)

                # Average log prob per sequence as a simple confidence metric
                confidences = token_log_probs.mean(dim=-1).exp().tolist() # .exp() converts Log-Prob to Prob
                decoded_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for ques, generation, conf in zip(batch_questions, decoded_generation, confidences):
                write_obj = {
                    'question': ques,
                    'generation': generation,
                    'avg_log_prob': round(conf, 4)  # Saving the confidence score
                }
                file.write(json.dumps(write_obj) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="2wikimultihopqa")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()
    main(model_name=args.model_name, dataset=args.dataset, batch_size=args.batch_size, gpu_id=args.gpu_id)