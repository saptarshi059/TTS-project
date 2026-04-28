from peft import PeftConfig

base_path = "/gpuhome/sks6765/.cache/huggingface/hub/models--yuchuanyue--GraphRAG-R1/snapshots/c03a3ddffc007b5cc1bf8c878b24445cf47edf3d/checkpoints/qwen_instruct_v2/"

config = PeftConfig.from_pretrained(base_path)
config.base_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
config.save_pretrained(base_path)