from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from urllib.parse import unquote
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import faiss
import json


def process_samples(contexts, tokenizer):
    total_chunks = []
    total_metadata = []
    for ctx in tqdm(contexts):
        chunks = tokenizer(ctx,
                           truncation=True,
                           max_length=400,
                           return_overflowing_tokens=True,
                           stride=50)['input_ids']
        chunks_detokenized = tokenizer.batch_decode(chunks, skip_special_tokens=True)

        for idx, chunk in enumerate(chunks_detokenized):
            total_chunks.append(chunk)
            total_metadata.append({
                "id": str(idx),
                "contents": chunk
            })

    return total_chunks, total_metadata


def build_index(model, texts, base_path, dataset_name):
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product on normalized vectors = Cosine Similarity
    index.add(np.array(embeddings).astype('float32'))
    print("Index created...")

    faiss.write_index(index, base_path / f'{dataset_name}_index.index')
    print("Index saved...")


def collect_contexts(dataset, dataset_name):
    all_contexts = []
    if dataset_name in {"2wikimultihopqa", "hotpotqa"}:
        for row in dataset.itertuples():
            row_ctx = row.context
            for ctx in row_ctx:
                title = ctx[0]
                text = ctx[1]
                if isinstance(text, str):
                    all_contexts.append(f"{title}\n{text}")
                else:
                    all_contexts.append(f"{title}\n{' '.join(text)}")
    elif dataset_name == "musique":
        for row in dataset.itertuples():
            row_ctx = row.paragraphs
            for ctx in row_ctx:
                title = ctx["title"]
                text = ctx["paragraph_text"]
                all_contexts.append(f"{title}\n{text}")
    else:
        for row in dataset.itertuples():
            ctx = row.Text
            title = unquote(row.URL.split('/')[-1])
            all_contexts.append(f"{title}\n{ctx}")

    return all_contexts


def main(dataset_name):
    print("Loading embedding model...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                model_kwargs={"attn_implementation": "flash_attention_2",
                                              "device_map": "auto",
                                              "dtype": "auto"},
                                tokenizer_kwargs={"padding_side": "left"}
                                )
    tokenizer = model.tokenizer
    print("Embedding model loaded...")

    base_path = Path(f"../../sampled_data/{dataset_name}")
    dataset = pd.read_json(base_path / "sampled_ds.json") if dataset_name != "frames" else pd.read_parquet(base_path / "frames_corpus")

    print("Collecting all contexts...")
    all_contexts = collect_contexts(dataset, dataset_name)

    print("Processing documents...")
    all_chunks, all_metadata = process_samples(all_contexts, tokenizer)

    print("Building Index...")
    build_index(model, all_chunks, base_path, dataset_name)

    with Path(base_path / f"{dataset_name}-chunks.jsonl").open("w") as file:
        json.dump(all_metadata, file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["2wikimultihopqa", "frames", "hotpotqa", "musique"])
    args = parser.parse_args()
    main(dataset_name=args.dataset)