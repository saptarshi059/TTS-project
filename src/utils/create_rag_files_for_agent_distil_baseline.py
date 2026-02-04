import json
from pathlib import Path

import faiss
from datasets import load_dataset, tqdm
from sentence_transformers import SentenceTransformer


def main():
    base_path = Path("../baselines/agent-distillation")
    all_datasets = ["2wikimultihopqa_500_20250511.json", "hotpotqa_500_20250422.json", "musique_500_20250504.json"]

    print("Loading wikipedia database...")
    wiki_corpus = load_dataset("json",
                               split="train",
                               num_proc=4,
                               data_files=str(base_path / "search/database/wikipedia/wiki-18.jsonl"))

    print("Loading Index...")
    cpu_index = faiss.read_index(str(base_path / "search/database/wikipedia/e5_Flat.index"))

    # 3. Transfer the existing index to the GPU
    # '0' refers to the GPU ID. If you have multiple GPUs, you can specify which one to use.
    print("Moving Index to GPUs...")
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    print("Index Loaded...")

    print("Loading embedding model...")
    model = SentenceTransformer("intfloat/e5-base-v2")

    for dataset in all_datasets:
        print(f"Working on {dataset}...")
        dataset_path = base_path / f"data_processor/qa_dataset/test/{dataset}"
        with Path(dataset_path).open("r") as f:
            questions_dataset = json.load(f)["examples"]

        # Have to append "query" for e5
        all_questions = [f"query: {ques}" for ques in questions_dataset]
        embedding_options = {"show_progress_bar": True, "convert_to_tensor": True}
        question_embeddings = model.encode(all_questions, **embedding_options)

        distances, indices = gpu_index.search(question_embeddings, 3) # Take top-3 docs

        all_retrieved_data = []

        # 1. Efficiently grab all documents at once. This is the 'heavy' part for the disk/RAM
        print("Fetching document text from dataset...")
        flat_indices = indices.flatten().tolist()
        retrieved_chunks = wiki_corpus.select(flat_indices)

        # 2. Process with a progress bar
        print("Assembling retrieved documents...")
        for q_idx in tqdm(range(len(all_questions)), desc="Processing Questions"):
            doc_entries = []

            for i in range(3):
                # Calculate the pointer for the flat retrieved_chunks
                pointer = (q_idx * 3) + i

                dist = distances[q_idx][i]
                doc_content = retrieved_chunks[pointer]["contents"]

                entry = f"\n\n===== Document {i}, Score: {dist:.2f} =====\n{doc_content}"
                doc_entries.append(entry)

            all_retrieved_data.append({all_questions[q_idx]: "\nRetrieved documents:" + "".join(doc_entries)})

        # 3. Save to JSON
        print(f"Saving to JSON...")
        with Path(base_path / f"data_processor/retrieved_documents/{dataset}.json").open("w", encoding="utf-8") as f:
            json.dump(all_retrieved_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()