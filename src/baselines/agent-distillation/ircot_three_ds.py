import faiss
import torch
from datasets import tqdm, Dataset, load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def load_everything():
    print("Loading wikipedia database...")
    corpus = load_dataset("json",
                          split="train",
                          num_proc=4,
                          data_files="wiki-18.jsonl")

    res = faiss.StandardGpuResources()

    print("Loading Index...")
    cpu_index = faiss.read_index("e5_Flat.index")

    # 3. Transfer the existing index to the GPU
    # '0' refers to the GPU ID. If you have multiple GPUs, you can specify which one to use.
    print("Moving Index to GPU...")
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    print("Index Loaded...")

    print("Loading embedding model...")
    model = SentenceTransformer("intfloat/e5-base-v2")

    print("Loading SLM...")
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                                      torch_dtype="auto",
                                                      device_map="auto",
                                                      attn_implementation="flash_attention_2")
    slm = PeftModel.from_pretrained(base_model, "agent-distillation/agent_distilled_Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    return corpus, gpu_index, model, slm, tokenizer

def run_ircot(question_list, corpus, index, embedding_model, slm, tokenizer):
    responses = {}
    for question in tqdm(question_list):
        collected_context = set()
        messages = [{"role": "system", "content": """Process the given query iteratively. At each step, produce one step and one search. 
Do not provide a final answer until you have all the information.

At each turn, generate, 
<step> The single immediate next sub-action required. </step>
<search> A single optimized query for this step. </search>

ONLY WHEN YOU HAVE ALL OF THE INFORMATION, generate,
<answer> Your final answer based on the supporting facts. Do NOT use this in TURN 0, unless you are absolutely certain. </answer>

Rules:
Stop immediately after the </search> tag.
No planning of future steps; focus only on the current one.
The <answer> tag is forbidden if any data is missing or if a search is still needed."""},
                    {"role": "user", "content": f"TURN 0\nQUESTION: {question}"}]

        for step in range(5):
            if step != 0:
                supporting_facts_string = "\n".join(collected_context)
                messages[1]["content"] = f"SUPPORTING FACTS: {supporting_facts_string}\nTURN {step}\nQUESTION: {question}"

            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([input_text], return_tensors="pt").to(slm.device)

            with torch.no_grad():
                generated_text = tokenizer.decode(slm.generate(**model_inputs, max_new_tokens=100)[0])
                generated_text = generated_text.split(input_text)[-1].strip()

            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
            search_match = re.search(r"<search>(.*?)</search>", generated_text, re.IGNORECASE | re.DOTALL)

            if answer_match:
                # Extract just the answer part after the colon
                responses[question] = answer_match.group(1).strip()
                break

            if search_match:
                thought = search_match.group(1).strip()
                _, new_docs_indices = index.search(embedding_model.encode([f"query: {thought}"], normalize_embeddings=True), k=3)
                for idx in new_docs_indices[0]:
                    if idx != -1:
                        doc = corpus[idx]["contents"]
                        if doc not in collected_context:
                            collected_context.add(doc)

        # If the model could not find an answer.
        if question not in responses:
            responses[question] = ""

    return responses

def main():
    wiki_corpus, index, embedding_model, slm, tokenizer = load_everything()
    all_datasets = ["2wikimultihopqa", "hotpotqa", "musique"]

    for dataset_name in tqdm(all_datasets):
        print(f"Working on {dataset_name}...")
        dataset = pd.read_json(f"../../../sampled_data/{dataset_name}/sampled_ds.json").sample(1)

        all_questions = [row.question for row in dataset.itertuples()]
        dataset_responses = run_ircot(all_questions, wiki_corpus, index, embedding_model, slm, tokenizer)
        print("Saving responses...")
        gold_answers = [row.answer for row in dataset.itertuples()]
        response_ds = Dataset.from_dict({"question": list(dataset_responses.keys()),
                                         "gold_answers": gold_answers,
                                         "response": list(dataset_responses.values())})
        response_ds.save_to_disk(f"{dataset_name}_responses")
        break

if __name__ == "__main__":
    main()
