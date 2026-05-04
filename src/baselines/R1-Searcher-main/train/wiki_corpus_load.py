from argparse import ArgumentParser
from pathlib import Path

import faiss
import pandas as pd
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

def load_corpus(file_path):
    return pd.read_json(file_path)['contents'].to_list()

app = Flask(__name__)

@app.route("/queries", methods=["POST"])
def query():
    data = request.json
    queries = data["queries"]
    k = data.get("k", 3)
    query_embeddings = model.encode(queries,  prompt_name="query")

    all_answers = []
    D, I = index.search(query_embeddings, k=k)
    for idx in I:
        answers_for_query = [corpus[i] for i in idx[:k]]
        all_answers.append(answers_for_query)

    return jsonify({"queries": queries, "answers": all_answers})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    data_type = args.dataset_name
    port = args.port

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                tokenizer_kwargs={"padding_side": "left"}
                                )

    base_path = Path("/gpuhome/sks6765/TTS-project/sampled_data")

    corpus_path = base_path / f"{args.dataset_name}/{args.dataset_name}-chunks.jsonl"
    corpus = load_corpus(corpus_path)

    index_path = f"{str(base_path)}/{args.dataset_name}/{args.dataset_name}_index.index"
    index = faiss.read_index(index_path)

    app.run(host="0.0.0.0", port=port, debug=False)
    print('Running server...')
