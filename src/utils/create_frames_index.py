from sentence_transformers import SentenceTransformer
from datasets import load_dataset, tqdm
from urllib.parse import unquote
from pathlib import Path
import wikipediaapi
import numpy as np
import faiss
import json
import ast

def main():
    def process_samples(links):
        total_chunks = []
        total_metadata = []
        for url in tqdm(links):
            page_name = unquote(url.split('/')[-1])
            page = wiki.page(page_name)
            if not page.exists():
                continue

            page_text = page.text
            chunks = tokenizer(page_text,
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

        faiss.write_index(index, "../baselines/agent-distillation/search/database/frames_wikipedia/e5_Flat.index")
        print("Index saved...")

    model = SentenceTransformer("intfloat/e5-base-v2")
    tokenizer = model.tokenizer

    dataset = load_dataset("google/frames-benchmark")['test']

    wiki = wikipediaapi.Wikipedia(
        user_agent="FramesBot/1.0 (contact: your@email.com)",
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    all_links = []
    for row in dataset:
        all_links.extend(ast.literal_eval(row['wiki_links']))
    all_chunks, all_metadata = process_samples(all_links)

    build_index(all_chunks)

    # keeping same name as agent-distillation, although it should be something like wiki-24
    with Path("../baselines/agent-distillation/search/database/frames_wikipedia/wiki-18.jsonl").open("w") as file:
        json.dump(all_metadata, file, indent=4)


if __name__ == "__main__":
    main()