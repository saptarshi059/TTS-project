import os
import argparse
import torch.distributed as dist
import json
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from openai import OpenAI
import sys
import os
import re
from datasets import load_dataset
import http.client
import json
import copy
from tqdm import tqdm
import multiprocessing
from time import sleep
import requests
from collections import defaultdict
import random
import requests
import time

# CUDA_VISIBLE_DEVICES=5



import re
from time import sleep
from openai import OpenAI
import requests
import json
import multiprocessing
from collections import defaultdict
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=100000)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--src_file", type=str, default="None")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--model_path", type=str, default="None")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.95)
    parser.add_argument("--port", type=str, default="None")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--prompt_type", type=str, default="None")
    return parser.parse_args()

def process_text(examples,tokenizer,type=None):

    base_prompt_v0 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

    base_prompt_v1 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step.
For each sub-question, **the Assistant can perform searching** for uncertain knowledge using the format: "<|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>".
**The query must consist of straightforward and essential keywords separated by "\t"**. Furthermore, **the query must involve only a single triple to address a sub-question**.
Then, the search system will provide the Assistant with relevant information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""
    base_prompt_v2="""The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

    base_prompt_v3 = """The User asks a **Judgment question**, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no) </answer>". During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". The final answer **must be yes or no**.\n\nUser:{question}\nAssistant: <think>"""

    if type == "v0":
        question = examples["question"]
        prompt = base_prompt_v0.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v1":
        question = examples["question"]
        prompt = base_prompt_v1.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v2":
        question = examples["question"]
        prompt = base_prompt_v2.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v3":
        question = examples["question"]
        prompt = base_prompt_v3.format(question=question)
        examples["chat_prompt"] = prompt
    else:
        kill
    return examples

# def process_text(examples,tokenizer,type=None):
#     sys_prompt_with_doc_token='''You are a helpful assistant. Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>". During the thinking process, you can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"'''
#     sys_prompt_wo_doc_token='''You are a helpful assistant. Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>". During the thinking process, you can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>".'''

#     question = examples["question"]
#     messages_chat_v1=[
#             {"role": "system","content":sys_prompt_wo_doc_token},
#             {"role": "user", "content":question}
#         ]
#     chat_prompt = tokenizer.apply_chat_template(
#                     messages_chat_v1,
#                     tokenize=False,
#                     add_generation_prompt=True
#                 )
#     examples["chat_prompt"] = chat_prompt + "<think>"
#     return examples

def main():
    print("=Begin="*10)
    args = parse_args()
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    temp=args.temp
    port=args.port
    type=args.prompt_type
    model_path=args.model_path
    gpu_memory_rate=args.gpu_memory_rate

    data_ori_all = []
    with open(args.src_file, "r") as f:
        data_ori_all = []
        for i, line in enumerate(f):
            if args.start_sample <= i < args.end_sample:
                obj_ori=json.loads(line)
                data_ori_all.append(obj_ori)
            if i >= args.end_sample - 1:
                break

    print("All Data Length: ",len(data_ori_all))
    chunk_size = 20000
    chunk_num = len(data_ori_all) // chunk_size
    if len(data_ori_all) % chunk_size != 0:
        chunk_num += 1
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=gpu_memory_rate, trust_remote_code=True)

    for h in range(chunk_num):
        print("=="*80)
        print("Begin Chunk: ",h,"All: ",chunk_num)
        data_ori = data_ori_all[h*chunk_size:(h+1)*chunk_size]
        data=[]

        for i in range(len(data_ori)):
            for j in range(1):
                data.append(data_ori[i])

        data_keys = data[0].keys()
        data_keys = ["question" , "answer"]
        ds = Dataset.from_dict({key: [d[key] for d in data] for key in data_keys})
        print(len(ds))
        ds = ds.map(
            process_text,
            num_proc=16,
            fn_kwargs={"tokenizer": tokenizer,"type":type},
        )
        print(ds)

        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
        sampling_params = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, stop=stop_tokens)

        finished_all_list=[]

        continued_answer = copy.deepcopy(data)

        for k in range(16):

            if len(ds) ==0:
                print("请确定是不是真的ok了")
                print(len(ds))
                break

            outputs = llm.generate(ds['chat_prompt'], sampling_params)

            finished_texts = []
            continued_texts = []

            gen_finished_texts = []
            query_list=[]

            for i, output in enumerate(outputs):

                prompt = output.prompt
                answer = continued_answer[i]["answer"]
                quesiton = continued_answer[i]["question"]
                gen_text_store = continued_answer[i]["gen_text_store"]
                stop_reason = output.outputs[0].stop_reason
                generated_text = output.outputs[0].text

                if k == 9: #检索次数太多了，直接停掉，就是未完成
                    original_data = {
                            "question":quesiton,
                            "answer": answer,
                            "generated_text":generated_text,
                            "stop_reason_final": "many_retrieve",
                            "pred_ans": "I don't know."
                        }

                    finished_texts.append(original_data)
                    continue

                if "<answer>" in generated_text and stop_reason=="</answer>":
                    original_data = {
                    "question":quesiton,
                    "answer": answer,
                    "pred_ans": generated_text.split("<answer>")[-1].split("</answer>")[0],
                    "stop_reason_final": "finished",
                    "gen_text_store": gen_text_store + generated_text + "</answer>",
                }
                    # gen_finished_texts.append(original_data)
                    finished_texts.append(original_data)

                elif "<|begin_of_query|>" in generated_text and stop_reason=="<|end_of_query|>": #这里处理retrieve
                    query = generated_text.split("<|begin_of_query|>")[-1].split("<|end_of_query|>")[0]
                    query = query.replace('"',"").replace("'","").replace("\t"," ").replace("...","")
                    if query:
                        topk = 5
                        gen_finished_texts.append(None)
                        query_list.append(query)

                        original_data = {
                            "chat_prompt":prompt + generated_text.strip(), #+ "<|end_of_query|> "+ "\n\nThe retrieved content are:\n<tool_call>\n"  +  doc_content + "\n</tool_call>\n\n",
                            "answer": answer,
                            "question":quesiton,
                            "stop_reason": stop_reason,
                            "gen_text_store": gen_text_store + generated_text.strip() #+ "<|end_of_query|> "+ "\n\nThe retrieved content are:\n<tool_call>\n"  +  doc_content + "\n</tool_call>\n\n",
                            }
                        continued_texts.append(original_data)
                    else:
                        original_data = {
                            "question":quesiton,
                            "answer": answer,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "generated_text":generated_text,
                            "stop_reason_final": "query_inst_error",
                            "pred_ans": "I don't know."
                        }
                        finished_texts.append(original_data)

                else:
                    original_data = {
                    "question":quesiton,
                    "answer": answer,
                    "stop_reason_final": "shot_down",
                    "pred_ans": "I don't know."
                }
                    finished_texts.append(original_data)

            print(query_list)
            print("=="*80)

            assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
            url_wiki = "http://0.0.0.0:"+port+"/queries"
            if len(query_list)!=0:
                response = requests.post(url_wiki, json={"queries": query_list, "k": topk})
                if response.status_code == 200:
                    result = response.json()
                    answers = result["answers"]
                    for i in range(len(answers)):
                        retrieve_docs = answers[i]
                        continued_text_now = copy.deepcopy(continued_texts[i])
                        if len(retrieve_docs)>0:
                            doc_content_list = []
                            for j in range(len(retrieve_docs)):
                                doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[j])
                                doc_content_list.append(f"({j+1}){doc_now}\n")
                            doc_content = ''.join(doc_content_list)

                        else:
                            doc_content = "None"
                        continued_text_now["chat_prompt"] = continued_text_now["chat_prompt"] + "<|end_of_query|>\n\n"+ "<|begin_of_documents|>\n" +  doc_content + "<|end_of_documents|>\n\n"
                        continued_text_now["gen_text_store"] = continued_text_now["gen_text_store"] + "<|end_of_query|>\n\n"+ "<|begin_of_documents|>\n" +  doc_content + "<|end_of_documents|>\n\n"
                        continued_texts[i] = continued_text_now

                else:
                    for i in range(len(continued_texts)):
                        current_data = continued_texts[i]  # 临时保存引用
                        original_data = {
                            "question": current_data["question"],
                            "answer": current_data["answer"],
                            "stop_reason_final": "retrieve_error",
                        }
                        continued_texts[i] = copy.deepcopy(original_data)
                    # raise Exception("Error in response: the status code is not 200!")

            finished_all_list.extend(finished_texts)

            if len(continued_texts)==0:
                if len(finished_texts)>0:
                    with open(args.src_file.replace(".jsonl","-"+model_path.split("/")[-2]+model_path.split("/")[-1]+f"_base_temp{args.temp}_type{type}.jsonl"), "a") as f:
                        for text in finished_texts:
                            f.write(json.dumps(text) + "\n")

                break
            else:
                data_keys_again = continued_texts[0].keys()
                ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                continued_answer = copy.deepcopy(continued_texts)
            print("=="*80)
            print("Epoch: ",k,"New_Finished: ",len(finished_texts),"All_Finished ",len(finished_all_list),"Continued: ",len(continued_texts))

            print("Begin Writing Epoch: ",k)

            # print(continued_texts)
            print("=="*80)
            # print(finished_texts)
            if len(finished_texts)>0:
                with open(args.src_file.replace(".jsonl","-"+model_path.split("/")[-2]+model_path.split("/")[-1]+f"_base_temp{args.temp}_type{type}.jsonl"), "a") as f:
                    for text in finished_texts:
                        f.write(json.dumps(text) + "\n")

    if dist.is_initialized():
            dist.destroy_process_group()
if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()

# python /opt/aps/workdir/sht-RAG_RL/eval/gen_ckpt_solution_base.py --src_file /opt/aps/workdir/sht-RAG_RL/eval/datasets/hotpotqa.jsonl --model_path /opt/aps/workdir/sht-RAG_RL/results/ckpts/qwen2.5-7B-base-rm3-sft-data-2-grpo-dataset_hpqa-len_29000-tbs_64-rbs_16-sample_16-kl_0.0001-warmup_0.0-ep_10000-plr_2e-6-temp1.0-30k/ckpt --gpu_id 0
