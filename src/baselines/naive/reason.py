from transformers import AutoModelForCausalLM, AutoTokenizer
import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    #    quantization_config=bnb
    )

    prompt = "Dambar Shah was a King of the Gorkha Kingdom, present-day Gorkha District, Nepal who reigned from 1636 until his death in 1645. He was the father of Krishna Shah. Krishna Shah was King of the Gorkha Kingdom in the Indian subcontinent, present-day Nepal, who succeeded his father Dambar Shah and reigned from 1645 till his death in 1661. He was the father of his successor Rudra Shah"

    # prepare the model input
    messages = [
        {"role": "system", "content": "Answer the given question without writing any additional text. Format your response as\nAnswer: <answer text>"},
        {"role": "user", "content": "Question: Who is the grandchild of Dambar Shah?"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

if __name__ == "__main__":
    main()