Source code for our paper _Can Hallucinations Be Useful? Solving Multi-Hop Questions With SLMs By
Chaining System-I/II Reasoning_.

> [!TIP] 
> Some models (e.g. BGE-large) may stall due to XET backend issues. For them, try `source src/utils/hf_env.sh` 

> [!NOTE] 
> ALL sampled datasets and indices can be downloaded from 
> `https://drive.google.com/drive/folders/1dsn6ky7HFN8piqjnXkGc1RjHis4NA1re?usp=sharing`. Use the `download_data.py`
> script for this. Keep the downloaded folder in the root of the repository.

The datasets were sampled and their indices were created using,
1. `src/utils/sample_datasets.py`
2. `src/utils/download_frames_corpus.py`
3. `src/utils/create_index.py` [We use Qwen3-0.6B as the embedding model]

Our framework can be run with Python 3.9. However, please solve the library versions as they can be a nightmare to
work out, but can be done of course 😅. At minimum, just have `pandas torch transformers sentence-transformers accelerate`
installed. I think that should be enough to run everything.

Our framework scripts are located in `src/framework`. To run everything, please follow the steps below,

1. **System-1**: `./system-1/run_all_system_1.sh` - This runs all system-1, creates parsed versions and, 
calculates results.
2. **System-2** (_Triple Generation_): `./system-2/triple_generation/run_all_triple_gen.sh` - This runs the triple
extraction pipeline and creates the necessary output files.
3. **System-2** (_Context Retrieval_): `./system-2/retrieval/run_all_retrieval.sh` - This performs the context retrieval
and creates the corresponding output file with the contexts.
4. **System-2** (_Final Generation_): `./system-2/triple_generation/generate_with_all_evidence.sh` - The last script
that needs to be run. It collects the retrieved contexts, the system-1 guess, the generated triples to provide the final
answer. Finally, it computes the results as well.

Final notes,
1. We also provide our sampled datasets so that others can easily build on our work and don't have to rerun our
experiments, although it is very easy to do so. Additionally, they can use the results in our paper, if using our
sampled datasets and index.

2. As the models are lightweight, it'll take at most 30 mins to run 
everything, if you wish to do so.