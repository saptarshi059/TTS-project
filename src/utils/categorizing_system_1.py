from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, DataCollatorWithPadding
from all_system_prompts import CATEGORIZATION_PROMPT
from torch.utils.data import Dataset, DataLoader
import torch, json, pandas as pd
from pathlib import Path
from tqdm import tqdm


class GenerationDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        messages = [{"role": "system", "content": CATEGORIZATION_PROMPT},
                    {"role": "user", "content": f"<input>\n"
                                                f"Question: {sample['question']}\n"
                                                f"Gold Answer: {sample['answer']}\n"
                                                f"Predicted Answer: {sample['system_1_guess']}\n"
                                                f"</input>"}]

        formatted_text = self.tokenizer.apply_chat_template(messages,
                                                            tokenize=False,
                                                            add_generation_prompt=True,
                                                            enable_thinking=False)
        model_inputs = self.tokenizer(formatted_text)

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
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=judge,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    print(f"Wrapping predictions dataframe with torch...")
    torch_dataset = GenerationDataset(tokenizer=tokenizer, dataset=combined_outputs_df)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=8, shuffle=False,
                                          collate_fn=DataCollatorWithPadding(tokenizer))

    with Path("streamed_responses.jsonl").open("a") as file:
        for batch in tqdm(torch_dataset_dataloader):
            with torch.no_grad():
                outputs = model.generate(**{k: v.to(model.device) for k, v in batch.items()}, max_new_tokens=50)
                # Get actual prompt lengths (accounting for padding)
                prompt_lengths = batch['attention_mask'].sum(dim=1)

                # Extract only generated tokens for each sample in the batch
                generated_ids = [
                    outputs[i, prompt_lengths[i]:]
                    for i in range(outputs.shape[0])
                ]

                # batch_decode handles list of tensors fine
                decoded_generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for idx, generation in enumerate(decoded_generations):
                write_obj = {'id': idx, 'generation': generation}
                file.write(json.dumps(write_obj) + '\n')
