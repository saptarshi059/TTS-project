import time
import os
import json
from urllib.parse import unquote

import pandas as pd
# from llama_index import SimpleDirectoryReader
import csv
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import yaml
from glob import glob
from itertools import chain
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from bs4 import BeautifulSoup
import random

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["bge-large-en-v1.5", "gtr-t5-xxl", "qwen3-Embedding-0.6B"],
    default="qwen3-Embedding-0.6B",
    help="Model to use",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="2wikimultihopqa",
    choices=["2wikimultihopqa", "hotpotqa", "musique", "frames"],
    help="Dataset to use",
)
parser.add_argument("--chunk_size", type=int, default=512, help="chunk size")
parser.add_argument("--chunk_overlap", type=int, default=0, help="chunk overlap")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
args = parser.parse_args()


def split_text(data):

    documents = []
    for record in data:
        if record["title"]:
            combined_text = record["title"] + "\n" + record["content"]
        else:
            combined_text = record["content"]
        documents.append(Document(text=combined_text))

    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    contents = [node.text for node in nodes]
    return contents


def build_index(embeddings, vectorstore_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, vectorstore_path)


if __name__ == "__main__":

    model = SentenceTransformer(config["model"][args.model],
                                model_kwargs={"attn_implementation": "flash_attention_2", "dtype": "auto", "device_map": "auto"},
                                tokenizer_kwargs={"padding_side": "left"})
    dataset_name = args.dataset
    vectorstore_path = f"../data/corpus/{dataset_name}/{dataset_name}.index"
    contents = []
    print("loading document ...")
    start = time.time()
    if dataset_name in {"2wikimultihopqa", 'hotpotqa'}:
        ds = pd.read_json(f"../../../../sampled_data/{dataset_name}/sampled_ds.json")

        data = {}
        for item in tqdm(ds.itertuples()):
            for title, sentences in item.context:
                para = " ".join(sentences)
                data[para] = title
        contents = [
            {"id": i, "content": text, "title": title}
            for i, (text, title) in enumerate(data.items())
        ]
    elif dataset_name == "musique":
        ds = pd.read_json("../../../../sampled_data/musique/sampled_ds.json")

        data = {}
        for item in tqdm(ds.itertuples()):
            for paragraph_dict in item.paragraphs:
                para = paragraph_dict["paragraph_text"]
                data[para] = paragraph_dict["title"]
        contents = [
            {"id": i, "content": text, "title": title}
            for i, (text, title) in enumerate(data.items())
        ]
    else: # FRAMES
        ds = pd.read_parquet("../../../../sampled_data/frames/frames_corpus")

        data = {}
        for item in tqdm(ds.itertuples()):
            para = item.Text
            data[para] = unquote(item.URL.split('/')[-1])
        contents = [
            {"id": i, "content": text, "title": title}
            for i, (text, title) in enumerate(data.items())
        ]

    contents = split_text(contents)
    embeddings = model.encode(contents, batch_size=600)
    with open(
        f"../data/corpus/{dataset_name}/chunk.json", "w", encoding="utf-8"
    ) as fout:
        json.dump(contents, fout, ensure_ascii=False)
    print("Building index ...")
    build_index(embeddings, vectorstore_path)
    end = time.time()
