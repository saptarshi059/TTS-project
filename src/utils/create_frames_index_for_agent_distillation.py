from sentence_transformers import SentenceTransformer
from datasets import load_from_disk, tqdm
from pathlib import Path
import numpy as np
import faiss
import json

def main():
    def process_samples(corpus_dataset):
        total_chunks = []
        total_metadata = []
        for row in tqdm(corpus_dataset):
            chunks = tokenizer(row['text'],
                               truncation=True,
                               max_length=400,
                               return_overflowing_tokens=True,
                               stride=100)['input_ids']
            chunks_detokenized = tokenizer.batch_decode(chunks, skip_special_tokens=True)

            for idx, chunk in enumerate(chunks_detokenized):
                total_chunks.append(f"passage: {chunk}")
                total_metadata.append({
                    "id": str(idx),
                    "contents": chunk
                })

        return total_chunks, total_metadata

    def build_index(texts):
        print(f"Encoding {len(texts)} chunks...")
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product on normalized vectors = Cosine Similarity
        index.add(np.array(embeddings).astype('float32'))
        print("Index created...")

        faiss.write_index(index, "../src/baselines/agent-distillation/frames_index.index")
        print("Index saved...")

    print("Loading embedding model...")
    model = SentenceTransformer("intfloat/e5-base-v2")
    tokenizer = model.tokenizer
    print("Embedding model loaded...")

    corpus = load_from_disk("../baselines/agent-distillation/frames_corpus")

    print("Processing documents...")
    all_chunks, all_metadata = process_samples(corpus)

    build_index(all_chunks)

    # keeping same name as agent-distillation, although it should be something like wiki-24
    with Path("../baselines/agent-distillation/frames-wiki.jsonl").open("w") as file:
        json.dump(all_metadata, file, indent=4)


if __name__ == "__main__":
    main()