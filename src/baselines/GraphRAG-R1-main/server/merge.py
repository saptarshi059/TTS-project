from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./qwen_instruct_v2"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_path, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(base_path)

print("Merging adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("qwen-merged")
tokenizer.save_pretrained("qwen-merged")