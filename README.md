> [!TIP] 
> Some models (e.g. BGE-large) may stall due to XET backend issues. For them, try `source src/utils/hf_env.sh` 

> [!NOTE] 
> ALL sampled datasets and indices can be downloaded from 
> `https://drive.google.com/drive/folders/1dsn6ky7HFN8piqjnXkGc1RjHis4NA1re?usp=sharing`. Use the `download_data.py`
> script for this. Keep the folder in the root of the repository.

The datasets were sampled and their indices were created using,
1. `src/utils/sample_datasets.py`
2. `src/utils/download_frames_corpus.py`
3. `src/utils/create_index.py` [We use Qwen3-0.6B as the embedding model]

