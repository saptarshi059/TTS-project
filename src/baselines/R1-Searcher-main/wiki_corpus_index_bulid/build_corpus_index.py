
import torch
import faiss
import pickle
import os
import numpy as np

all_corpus_embeddings = []
first_tensors = []
for i in range(14):
    # i=j+1
    # path = f"resources/data/corpus/nq/para_8part/part-0{i}-corpus_embedding.pickle"
    path = f"/opt/aps/workdir/model/kilt_100/split_{i}.pickle"
    # path = f"/opt/aps/workdir/sht-RAG_RL/train/wiki_server/data/enwiki_add_2wiki.pickle"
    with open(path, 'rb') as f:
        ce = pickle.load(f)
        # ce = ce.to('cpu')
        all_corpus_embeddings.append(ce)
        # print(len(all_corpus_embeddings))
        first_tensor = ce[0]
        first_tensors.append(first_tensor)
        print(path)
        print(ce[0:1])
        print("=="*20)
        # kill
        print(f"Load corpus embeddings from {path} with shape of {ce.shape}.")
first_tensors = [torch.tensor(tensor) if isinstance(tensor, np.ndarray) else tensor for tensor in first_tensors]
are_same = all(torch.equal(first_tensors[0], tensor) for tensor in first_tensors)
if are_same:
    print("All first tensors are the same.")
else:
    print("The first tensors are not the same.")

corpus_embeddings = np.concatenate(all_corpus_embeddings, axis=0)
print(f"Cat all corpus embeddings with shape of {corpus_embeddings.shape}.")

dim = corpus_embeddings.shape[-1]
index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
print(index.is_trained)
index.add(corpus_embeddings)
print(f"total number of vectors: {index.ntotal}")

path = "/opt/aps/workdir/model/kilt_100/enwiki_kilt_all.bin"
# path = "/media/jiangjinhao/RAG-Star/enwiki-abs-index_w_title-bge-large-en-v1.5.bin"
faiss.write_index(index, path)

