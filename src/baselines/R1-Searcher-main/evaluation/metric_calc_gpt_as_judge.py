from openai import OpenAI
import sys
import os
from datasets import load_dataset
import http.client
import json
from tqdm import tqdm
import multiprocessing
from time import sleep
import requests
import json
from collections import defaultdict
import random
import json
import requests
import time
# from bs4 import BeautifulSoup
# import wikipediaapi
from urllib.parse import unquote
from urllib.parse import urlparse


os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["OPENAI_API_BASE"] = "xxx"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)


def generate(messages, model_name):
    response = client.chat.completions.create(
        **{
            "model": model_name,
            "messages": messages,
            "max_tokens": 1024,
        }
    )
    response = response.choices[0].message.content
    return response

def process_one_sample(obj):
#     prompt = """You will receive a question along with a reference answer and a predicted answer. Your task is to evaluate the accuracy of the predicted answer and provide a concise explanation.

# Compare the predicted answer to the reference answer to determine its correctness.

# **Guidelines**
# - The criteria for evaluating the predicted answer should not be overly strict. If the predicted answer's meaning aligns closely with that of the reference answer, it can be deemed correct.
# - For each question, provide a brief explanation of your reasoning, followed by "Correct" or "Incorrect." Include your final assessment within <assessment> tags.

# **Output Format**
# [Explanation]: Provide a brief explanation supporting your judgment.
# [Assessment]: Provide your assessment **within <assessment> tags**.

# Here is the question:
# {question}

# Here is the reference answer:
# {reference}

# Here is the predicted answer:
# {prediction}
# """
    prompt = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.

Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}
    '''

    question = obj["question"]
    reference_ans = obj["answer"]
    if reference_ans ==False:
        reference_ans="no"
    if reference_ans ==True:
        reference_ans="yes"
    # solution = obj["solution"]

    flag_final_ans = False

    # for k, line in enumerate(lines):
    #     if "oxed{" in line:
    # proposed_ans = solution.split("<answer>")[-1].split("</answer>")[0]
    proposed_ans = obj["pred_ans"]

    gpt4o_input = prompt.format(question=question , reference=reference_ans, prediction=proposed_ans)
    flag_final_ans = True

    if flag_final_ans:
        messages = [{'role': 'user', 'content': gpt4o_input}]
        model_output = generate(messages, 'gpt-4o-mini')
        obj["gpt4o_output"] = model_output
    else:
        obj["gpt4o_output"] = "！！ No boxed here ！！"

    try:
        print("=="*70)
        print(question)
        print(proposed_ans)
        print(reference_ans)
        print(obj["gpt4o_output"])
        print("=="*70)
    except:
        print("**"*40)
        print(question)
        print("**"*40)


    return obj



if __name__ == '__main__':
    input_files=[

    "/opt/aps/workdir/sht-RAG_RL/eval/datasets/musique_subset/qwen_zero.jsonl",
    ]
    # base_path="/home/songhuatong/RAG-Long-Cot/sys_data/data/a_seed_sft_new_new_new/origianl_data"
    # files= os.listdir(base_path)
    # input_files = [base_path+"/"+file for file in files]

    for input_file in input_files:
        output_file = input_file.replace(".jsonl","_judge.jsonl")
        chunk_size=400

        with open(input_file,"r") as fin:
            all_demons = fin.readlines()
            all_demons = [json.loads(s) for s in all_demons]

        print("All Data Num:",len(all_demons))
        chunk_num = len(all_demons) // chunk_size
        if len(all_demons) % chunk_size != 0:
            chunk_num += 1

        for chunk_i in range(chunk_num):
            print("Epoch:" ,chunk_i ,"/",chunk_num)
            all_demons_subset = all_demons[chunk_i*chunk_size : (chunk_i+1)*chunk_size]
            print(len(all_demons_subset))
            with multiprocessing.Pool(processes=400) as pool:
                results = list(tqdm(pool.imap(process_one_sample, all_demons_subset), total=len(all_demons_subset)))

            with open(output_file, 'a') as fout:
                for res in results:
                    if res is not None:
                        fout.write(json.dumps(res)+'\n')


