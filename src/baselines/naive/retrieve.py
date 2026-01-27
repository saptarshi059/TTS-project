from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

def main(dataset_name: str, batch_size: int) -> None:
    base_path = Path(f"../../../data/{dataset_name}")

    # Loading Graph associated with the dataset
    with open(base_path / "kg.txt", "r") as f:
        kg = f.read().split("\n")

    # Loading test questions for the dataset
    dataset = load_dataset("json", data_files=str(base_path / "test.json"))["train"]
    all_questions = dataset["question"]

    # Loading embedding model
    model = SentenceTransformer(model_name_or_path="Qwen/Qwen3-Embedding-8B",
                                model_kwargs={"attn_implementation": "flash_attention_2"})

    print("Creating embeddings .....")
    embedding_options = {"show_progress_bar": True, "convert_to_tensor": True, "batch_size": batch_size}

    kg_triple_embeddings = model.encode(kg, **embedding_options)
    query_embeddings = model.encode(all_questions, **embedding_options)

    hits = semantic_search(query_embeddings, kg_triple_embeddings, top_k=10)

    retrieved_triples = []
    for hit_list in tqdm(hits):
        retrieved_triples.append([kg[x['corpus_id']] for x in hit_list])

    dataset.add_column("retrieved_triples", retrieved_triples)

    print("Saving dataset with retrieved triples...")
    dataset.save_to_disk(base_path / f"{dataset_name}_with_retrieved_triples_from_naive_baseline")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    args = parser.parse_args()

    main(dataset_name=args.dataset_name, batch_size=args.batch_size)