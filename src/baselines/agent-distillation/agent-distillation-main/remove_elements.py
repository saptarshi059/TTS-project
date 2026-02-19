import pandas as pd

path = '/gpuhome/sks6765/.cache/huggingface/hub/models--agent-distillation--agent_distilled_Qwen2.5-7B-Instruct/snapshots/816cf2f90baa7948ddb29cd0667b1d83567b0707/qa_results/2wikimultihopqa_test/a09a35458c702b33eeacc393d103063234e8bc28_temp=0.4_n=1_seed=42_type=agent_steps=5.jsonl'

s = pd.read_json(path, lines=True)
s = s[s['error'] != 'Error in generating model output:\nConnection error.']
s.to_json(path, orient='records', lines=True)