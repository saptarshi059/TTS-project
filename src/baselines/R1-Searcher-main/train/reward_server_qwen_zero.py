import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import datasets
import random
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer
# from symeval import EvaluatorMathBatch
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

logger = init_logger(__name__)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s

def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )

def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    # 不考虑顺序和连续
    return all(ground in pre_list for ground in ground_list)

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    return s.split("<answer>")[-1].split("</answer>")[0].strip()

def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()


class MathRuleProxy:
    def __init__(self, args):
        eval_dataset = datasets.load_from_disk(args.data_path).to_list()
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        print(len(self.eval_data_dict))
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True, use_fast=True)
        self.log_file = args.log_file
        self.avg_length_dict = []
        self.cnt = 0
        self.avg_len = 5000
        self.key_words = [
            "wait",
            "double check",
            "what",
            "how",
            "why",
            "alternatively",
            "think",
            "rethink",
            "?",
            "change",
            "try",
            "check",
        ]

    def get_answer_dict(self, eval_dataset):
        eval_data_dict = {}
        for item in eval_dataset:
            eval_data_dict[normalize_text(item["question"])] = item["answer"]
        return eval_data_dict

    def get_qa(self, query):
        remove_prefix = " ".join(query.split("\n\nUser:")[1:])
        question = remove_prefix.split("\nAssistant: <think>")[0].strip()
        solution = query.split("\nAssistant: <think>")[-1].strip()
        return question, solution

    def get_query_answer(self, query):
        query = normalize_text(query)
        return self.eval_data_dict[query]

    def get_query_pred(self, query):
        return extract_answer_math(query)

    def get_reward(self, queries):
        preds = []
        answers = []
        questions = []
        solutions = []
        finished_lst = []
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
            print(queries[i])
            question, solution = self.get_qa(queries[i])
            preds.append(self.get_query_pred(solution))
            answers.append(self.get_query_answer(question))

            questions.append(question)
            solutions.append(solution)
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        for t in range(len(queries)):
            f1_score_now, _ , _ = f1_score(preds[t], answers[t])
            scores.append(float(f1_score_now))

        length_scores = []
        pattern_scores = []
        for i, query in enumerate(queries):
            self.cnt = self.cnt + 1
            if "<answer>" not in solutions[i] or "</answer>" not in solutions[i]:
                scores[i] = 0.0
                finished_lst.append("0")
            else:
                finished_lst.append("1")

            format_punishment=False
            count_1 = solutions[i].count("<|begin_of_documents|>\n")
            count_2 = solutions[i].count("<|end_of_documents|>\n\n")
            count_3 = solutions[i].count("<|begin_of_query|>")
            count_4 = solutions[i].count("<|end_of_query|>")
            count_5 = solutions[i].count("<|begin_of_documents|>")
            count_6 = solutions[i].count("<|end_of_documents|>")
            count_7 = solutions[i].count("<|begin_of_documents|>\n(1)")

            if count_1 == count_2 == count_3 == count_4 == count_5 == count_6 == count_7:
                pass
            else:
                format_punishment=True

            count_assiatant_1 = solutions[i].count("Assistant")
            count_assiatant_2 = solutions[i].count("assistant")
            if count_assiatant_1 == count_assiatant_2 ==0:
                pass
            else:
                format_punishment=True

            count_think_1 = solutions[i].count("<think>")
            count_think_2 = solutions[i].count("</think>")
            if count_think_1 ==0 and count_think_2==1:
                pass
            else:
                format_punishment=True

            count_answer_1 = solutions[i].count("<answer>")
            count_answer_2 = solutions[i].count("</answer>")
            if count_answer_1 == count_answer_2==1:
                pass
            else:
                format_punishment=True

            answer_text = solutions[i].split("<answer>")[-1].split("</answer>")[0].strip()
            if "begin_of_query" not in answer_text and "begin_of_documents" not in answer_text:
                pass
            else:
                format_punishment=True

            answer_len=len(answer_text.split())
            if answer_len > 10:
                format_punishment=True

            modified_solution = re.sub(r'<\|begin_of_documents\|>.*?<\|end_of_documents\|>', '', solutions[i], flags=re.DOTALL)
            have_chinese = any('\u4e00' <= char <= '\u9fff' for char in modified_solution)
            if have_chinese:
                format_punishment=True


            if format_punishment==True:
                scores[i] = scores[i]-2

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for q, a, s, f_f in zip(
                    questions,
                    solutions,
                    scores,
                    finished_lst,
                ):
                    record = {
                        "question": q,
                        "solution": a,
                        "score": s,
                        "finished": f_f,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--log_file", type=str, default=None, help="Path to JSONL log file")

    args = parser.parse_args()

    # server
    reward_model = MathRuleProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        # print("sht-debug-"*30)
        # print(data)
        # print("sht-debug-"*30)
        queries = data.get("query")
        # print(queries)
        # print("sht-debug-"*30)
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# python /home/songhuatong/OpenRLHF/openrlhf/cli/server_rm_rag.py --data_path /home/songhuatong/OpenRLHF/data/hotpotqa_rollout_10 --reward_pretrain /home/songhuatong/Qwen2.5-1.5B-Instruct --log_file /home/songhuatong/RAG_RL/rewards/sampling.jsonl --port 1278 --host 127.0.0.1