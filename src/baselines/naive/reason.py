from datasets import tqdm, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
import os, torch
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NaiveDataset(Dataset):
    def __init__(self, tokenizer, dataset, device):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device

        self.samples = []
        for row in tqdm(dataset):
            supporting_facts = "\n".join(triple for triple in row['retrieved_triples'])
            self.samples.append([{"role": "system", "content": "Answer the given question using the provided knowledge graph triples. Format your response as\nAnswer: <answer text>"},
                                 {"role": "user", "content": f"SUPPORTING FACTS:\n{supporting_facts}\n\nQUESTION: {row['question']}"}])

        self.tokenized_samples = tokenizer.apply_chat_template(
            self.samples,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )

        self.model_inputs = self.tokenizer(self.tokenized_samples, padding=True, return_tensors="pt")

    def __len__(self):
        return len(self.model_inputs)

    def __getitem__(self, idx):
        sample = self.model_inputs[idx]
        input_ids = sample["input_ids"].to(self.device)
        attention_mask = sample["attention_mask"].to(self.device)
        return input_ids, attention_mask


def main(model_name:str, dataset_path:str, batch_size: int) -> None:

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 dtype="auto",
                                                 attn_implementation="flash_attention_2",
                                                 device_map="auto")

    main_dataset = load_from_disk(dataset_path)
    torch_dataset = NaiveDataset(tokenizer=tokenizer, dataset=main_dataset, device=model.device)
    torch_dataset_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    generated_answers = []

    for batch in tqdm(torch_dataset_dataloader):
        with torch.no_grad():
            generated_ids = model.generate(input_ids=batch[0], attention_mask=batch[1], max_new_tokens=50)
            break



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset_path", type=str, default="../../../data/2wiki/2wiki_with_retrieved_triples_from_naive_baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(model_name=args.model_name, dataset_path=args.dataset_path, batch_size=args.batch_size)