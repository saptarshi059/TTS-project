MODEL_PATH="/gpuhome/sks6765/.cache/huggingface/hub/models--inclusionAI--ASearcher-Local-7B/snapshots/bd1b05b86ca7fae5617c608008a57e12e592a8b2"
DATA_DIR="/gpuhome/sks6765/TTS-project/sampled_data"

DATA_NAMES="2wikimultihopqa"
AGENT_TYPE="asearcher"
PROMPT_TYPE="local-rag"
SEARCH_CLIENT_TYPE="async-search-access"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
CUDA_VISIBLE_DEVICES="0,1" \
TOKENIZERS_PARALLELISM=false \
python3 search_eval_async.py \
    --data_names ${DATA_NAMES} \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --prompt_type ${PROMPT_TYPE} \
    --agent-type ${AGENT_TYPE} \
    --search_client_type ${SEARCH_CLIENT_TYPE} \
    --tensor_parallel_size 2 \
    --temperature 0.6 \
    --parallel-mode seed \
    --seed 1 \
    --pass-at-k 1