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
import os
from openai import OpenAI
import sys
import os
from datasets import load_dataset
import http.client
import json
from tqdm import tqdm
import multiprocessing
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import json
from collections import defaultdict
import random
import json
import requests
import time
import re
import concurrent.futures
from bs4 import BeautifulSoup
# import wikipediaapi
import wikipedia
from urllib.parse import unquote
from urllib.parse import urlparse

import json
import requests
import time
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
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
# import wikipediaapi
import wikipedia
from urllib.parse import unquote
from urllib.parse import urlparse
# CUDA_VISIBLE_DEVICES=5
import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
# import wikipediaapi


import re
from time import sleep
from openai import OpenAI
import requests
import json
import multiprocessing
from collections import defaultdict
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}
summary_prompt='''## Task Description:
Given the search query and the content of the searched webpage.
Your task is to extract information from the webpage content that is relevant to the search query and return a summary paragraph.

## **Guidelines**:
(1) The extracted content should be relevant to the query.
(2) The form of the extracted content **must be a summary paragraph** rather than a direct answer to the query.
(3) You **must extract content according to this webpage**. If the webpage content is unrelated to the query, no extraction is required.

## Output Format:
[Exacted Content]: If the webpage content contains information related to the query, output the relevant summary paragraph (not a direct answer to the query); if not, output "None".

## Inputs:
[Search Query]
{search_query}

[Webpage Content]
{document}

## Output:
'''
proxies = {
                "http": "http://127.0.0.1:7880",
                "https": "http://127.0.0.1:7880",
            }
# Initialize session
session = requests.Session()
session.headers.update(headers)
session.proxies.update(proxies)
wikipedia.set_lang('en')  # 这里设置为你需要的语言
wikipedia._http = session  # 替换为自定义的 session

os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["OPENAI_API_BASE"] = "xxx"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):

    try:
        if "wikipedia.org" in url:
            wiki_200=0
            for u in range(3):
                try:
                    print(url)
                    # print(unquote(url.split('/')[-1]))
                    page_title = unquote(url.split('/')[-1])
                    # print("000")
                    page = wikipedia.page(page_title, auto_suggest=False)
                    # print("111")
                    text = page.content
                    text = text.replace('\n\n', '\n')
                    search_doc = text.split('== References ==')[0].split("== Notes ==")[0].strip()
                    wiki_200=1
                    # print("222")
                    return search_doc
                except:
                    time.sleep(1)
                    continue
            if wiki_200==0:
                search_doc = "None"
                print("无法访问该页面，url:",url)
                return search_doc
        else:
            flag_200=0
            for t in range(3):
                try:
                    response = session.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text,'lxml')
                        text_before_summary = soup.get_text(separator='\n', strip=True)
                        lines_before_summary=text_before_summary.split("\n")
                        search_doc=" ".join(line for line in lines_before_summary if len(line.split(' '))>=3)
                        flag_200=1
                        return search_doc
                except:
                    sleep(2)
                    continue

            if flag_200==0:
                search_doc = "None"
                print("无法访问该页面，状态码：", response.status_code, "url:",url)
                kill
    except:
        pass



def bing_web_search(query, subscription_key, endpoint, market='en-US', language='en', timeout=20):

    payload = json.dumps({
        "q": query,  # 设置查询内容
        "mkt": market,  # 设置市场
        "setLang": language,  # 设置语言
        "textDecorations": True,  # 启用文本装饰
        "textFormat": "HTML"  # 设置文本格式
    })

    headers = {
        'X-API-KEY': subscription_key,
        'Content-Type': 'application/json'
    }

    try:
        # 发送POST请求
        response = requests.request("POST", endpoint, headers=headers, data=payload)
        response.raise_for_status()  # Raise exception if the request failed 检查响应的状态码。如果返回的状态码是 4xx 或 5xx（表示客户端或服务器错误），它将引发 requests.exceptions.HTTPError 异常
        search_results = response.json() #
        return search_results
    except Timeout:
        print(f"Bing Web Search request timed out ({timeout} seconds) for query: {query}")
        return {}  # Or you can choose to raise an exception
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Bing Web Search request: {e}")
        return {}



def extract_relevant_info(search_results):

    useful_info = []

    if 'organic' in search_results : # value 通常是一个列表，包含了搜索结果的每个页面信息
        for id, result in enumerate(search_results['organic']):
            # if "wikipedia.org" not in result.get('link', ''):
            #     continue
            info = {
                'title': result.get('title', ''), # 每个搜索结果中提取标题
                'url': result.get('link', ''), # 每个搜索结果中提取 URL
            }
            useful_info.append(info)

    return useful_info

def generate(messages, model_name):
    response = client.chat.completions.create(
        **{
            "model": model_name,
            "messages": messages,
            "max_tokens": 2048,
        }
    )
    response = response.choices[0].message.content
    return response

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

# def process_text(examples,tokenizer,type=None):

#     base_prompt_v0 = """The User asks a question, and the Assistant solves it.
# The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
# The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
# During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>".
# The proposed query must search for a straightforward sub-question. Furthermore, **the query must involve ONLY a single triple**.
# Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

#     base_prompt_v1 = """The User asks a question, and the Assistant solves it.
# The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
# The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
# During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step.
# For each sub-question, **the Assistant can perform searching** for uncertain knowledge using the format: "<|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>".
# **The query must consist of straightforward and essential keywords separated by "\t"**. Furthermore, **the query must involve only a single triple to address a sub-question**.
# Then, the search system will provide the Assistant with relevant information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

# User:{question}
# Assistant: <think>"""
#     base_prompt_v2="""The User asks a question, and the Assistant solves it.
# The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
# The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
# During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
# Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

# User:{question}
# Assistant: <think>"""

#     base_prompt_v3 = """The User asks a **Judgment question**, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no) </answer>". During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". The final answer **must be yes or no**.\n\nUser:{question}\nAssistant: <think>"""

#     if type == "v0":
#         question = examples["question"]
#         prompt = base_prompt_v0.format(question=question)
#         examples["chat_prompt"] = prompt
#     elif type=="v1":
#         question = examples["question"]
#         prompt = base_prompt_v1.format(question=question)
#         examples["chat_prompt"] = prompt
#     elif type=="v2":
#         question = examples["question"]
#         prompt = base_prompt_v2.format(question=question)
#         examples["chat_prompt"] = prompt
#     elif type=="v3":
#         question = examples["question"]
#         prompt = base_prompt_v3.format(question=question)
#         examples["chat_prompt"] = prompt
#     else:
#         kill
#     return examples

def process_text(examples,tokenizer,type=None):
    question = examples["question"]

    # sys_prompt_with_doc_token='''You are a helpful assistant. Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>". During the thinking process, you can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"'''
    # sys_prompt_wo_doc_token='''You are a helpful assistant. Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>". During the thinking process, you can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>".'''

    # messages_chat_v1=[
    #         {"role": "system","content":sys_prompt_wo_doc_token},
    #         {"role": "user", "content":question}
    #     ]
    if type=="v3":
        messages_chat=[
                {"role": "system","content": """You are a helpful assistant.
    Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
    The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
    During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
    Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""},
                {"role": "user", "content":question}
            ]

    elif type=="v0":
        messages_chat=[
                {"role": "system","content": """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>".
The proposed query must search for a straightforward sub-question. Furthermore, **the query must involve ONLY a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""},
                {"role": "user", "content":question}
            ]
    elif type=="v2":
        messages_chat=[
                {"role": "system","content": """You are a helpful assistant. Given a **Judgement question**, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no)</answer>". During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". The final answer **must be yes or no**."""},
                {"role": "user", "content":question}
            ]
    else:
        kill


    chat_prompt = tokenizer.apply_chat_template(
                    messages_chat,
                    tokenize=False,
                    add_generation_prompt=True
                )
    examples["chat_prompt"] = chat_prompt + "<think>"
    return examples

def process_output(output, continued_answer, k):
    prompt = output.prompt
    answer = continued_answer["answer"]
    quesiton = continued_answer["question"]
    gen_text_store = continued_answer["gen_text_store"]
    stop_reason = output.outputs[0].stop_reason
    generated_text = output.outputs[0].text

    # Return 'finished' or 'continued' along with the corresponding data
    if k == 8:  # 检索次数太多了，直接停掉，就是未完成
        original_data = {
            "question": quesiton,
            "answer": answer,
            "generated_text": generated_text,
            "stop_reason_final": "many_retrieve",
            "pred_ans": "I don't know."
        }
        return original_data, "finished"

    if "<answer>" in generated_text and stop_reason == "</answer>":
        original_data = {
            "question": quesiton,
            "answer": answer,
            "pred_ans": generated_text.split("<answer>")[-1].split("</answer>")[0],
            "stop_reason_final": "finished",
            "gen_text_store": gen_text_store + generated_text + "</answer>",
        }
        return original_data, "finished"

    elif "<|begin_of_query|>" in generated_text and stop_reason == "<|end_of_query|>":  # 这里处理retrieve
        query = generated_text.split("<|begin_of_query|>")[-1].split("<|end_of_query|>")[0]
        query = query.replace('"', "").replace("'", "").replace("\t", " ").replace("...", "")

        if query:
            # Reuse search and info extraction logic
            BING_SUBSCRIPTION_KEY = "xxx"
            bing_endpoint = "xxx"
            search_results = bing_web_search(query + " site:en.wikipedia.org", BING_SUBSCRIPTION_KEY, bing_endpoint)
            extracted_info = extract_relevant_info(search_results)

            doc_content = "None"
            for info in extracted_info[:3]:
                print("Begin Get Full Page")
                full_text = extract_text_from_url(info['url'])
                # print(query)
                # print(full_text)
                print("End Get Full Page")
                search_doc = full_text[:35000]
                query_summary = query
                summary_for_gpt = summary_prompt.replace("{search_query}", query_summary).replace("{document}", search_doc)
                messages_summary = [{'role': 'user', 'content': summary_for_gpt}]
                # print("messages_summary:",messages_summary)

                model_output_summary = generate(messages_summary, 'gpt-4o-mini')
                # print("model_output_summary",model_output_summary)
                # kill
                summary_doc = model_output_summary.split("[Exacted Content]")[-1]
                doc_content = summary_doc.lstrip(":").strip()
                print("End Summarize")
                if "none" not in summary_doc.lower():
                    break

                # print("=="*40)
                # print(query)
                # print(extracted_info)
                # print("=="*40)

            original_data = {
                "chat_prompt": prompt + generated_text.strip() + "<|end_of_query|>\n\n" + "<|begin_of_documents|>\n" + doc_content + "\n<|end_of_documents|>\n\n",
                "answer": answer,
                "question": quesiton,
                "stop_reason": stop_reason,
                "gen_text_store": gen_text_store + generated_text.strip() + "<|end_of_query|>\n\n" + "<|begin_of_documents|>\n" + doc_content + "\n<|end_of_documents|>\n\n",
            }
            return original_data, "continued"
        else:
            original_data = {
                "question": quesiton,
                "answer": answer,
                "gen_text_store": gen_text_store + generated_text.strip(),
                "generated_text": generated_text,
                "stop_reason_final": "query_inst_error",
                "pred_ans": "I don't know."
            }
            return original_data, "finished"

    else:
        original_data = {
            "question": quesiton,
            "answer": answer,
            "stop_reason_final": "shot_down",
            "pred_ans": "I don't know."
        }
        return original_data, "finished"

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
    chunk_size = 5
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
            with ThreadPoolExecutor() as executor:
                futures = []
                for i, output in enumerate(outputs):
                    futures.append(executor.submit(process_output, output, continued_answer[i], k))

                for future in as_completed(futures):
                    obj, label = future.result()
                    if label == "finished":
                        finished_texts.append(obj)
                    elif label == "continued":
                        continued_texts.append(obj)

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

