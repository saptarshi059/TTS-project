import faiss
import torch
from datasets import load_dataset, tqdm, Dataset
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
        messages = [{"role": "system",
                     "content": """
You are an expert assistant who can answer the given question accurately and provide clear reasoning.
If you cannot answer the question, then you can request for more information by formatting your output with <search> xxx </search>. You can ONLY generate 1 search request.
If you cannot find the needed information, use a different search request. 
If the question requires multiple steps or facts, break down your reasoning accordingly.
ONLY when you have the needed information say <answer> xxx </answer>. 
Respond with either <search> or <answer> but NOT BOTH. 

EXAMPLE:
QUESTION: When was the last time Brazil won the FIFA world cup?
<search> 
Brazil world cup records.
</search>
SUPPORTING FACTS: Brazil won in 1958, 1962, 1970, 1994, 2002.
<answer>
Brazil won last time in 2002.
</answer>
"""},
                    {"role": "user", "content": f"QUESTION: {question}"}]

        for step in range(5):
            if step != 0:
                supporting_facts_string = "\n".join(collected_context)
                messages[1]["content"] = f"SUPPORTING FACTS: {supporting_facts_string}\nQUESTION: {question}"

            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([input_text], return_tensors="pt").to(slm.device)

            with torch.no_grad():
                generated_text = tokenizer.decode(slm.generate(**model_inputs, max_new_tokens=100)[0])
                generated_text = generated_text.split(input_text)[-1].strip()

            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.IGNORECASE | re.DOTALL)
            search_match = re.search(r"<search>(.*?)</search>", generated_text, re.IGNORECASE | re.DOTALL)

            print(messages, "\n-------------------")
            print(generated_text, "\n................")
            print(search_match, "\n***************************")

            if answer_match:
                # Extract just the answer part after the colon
                responses[question] = answer_match.group(1).strip()
                break

            if search_match:
                thought = search_match.group(1).strip()
                print(f"Searching for documents related to: {thought}")
                _, new_docs_indices = index.search(embedding_model.encode([f"query: {thought}"], normalize_embeddings=True), k=3)
                for idx in new_docs_indices[0]:
                    if idx != -1:
                        doc = corpus[idx]["contents"]
                        if doc not in collected_context:
                            collected_context.add(doc)

        # If the model could not find an answer.
        if question not in responses:
            print("Model could not find an answer...")
            responses[question] = ""

    return responses

def main():
    wiki_corpus, index, embedding_model, slm, tokenizer = load_everything()
    all_datasets = ["2wikimultihopqa", "hotpotqa", "musique"]

    for dataset_name in tqdm(all_datasets):
        print(f"Working on {dataset_name}...")
        dataset = load_dataset('json',
                               data_files=f"../../../data/{dataset_name}/test.json",
                               split='train').select(range(2))

        all_questions = [row['question'] for row in dataset]
        dataset_responses = run_ircot(all_questions, wiki_corpus, index, embedding_model, slm, tokenizer)
        print("Saving responses...")
        gold_answers = [row['answer'] for row in dataset]
        response_ds = Dataset.from_dict({"question": list(dataset_responses.keys()),
                                         "gold_answers": gold_answers,
                                         "response": list(dataset_responses.values())})
        response_ds.save_to_disk(f"{dataset_name}_responses")
        break

if __name__ == "__main__":
    main()
