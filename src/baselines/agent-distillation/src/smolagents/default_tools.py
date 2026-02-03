#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)
from .tools import PipelineTool, Tool


@dataclass
class PreTool:
    name: str
    inputs: Dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class PythonInterpreterTool(Tool):
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter",
        }
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, **kwargs):
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))
        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {self.authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        state = {}
        output = str(
            self.python_evaluator(
                code,
                state=state,
                static_tools=self.base_python_tools,
                authorized_imports=self.authorized_imports,
            )[0]  # The second element is boolean is_final_answer
        )
        return f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer


class UserInputTool(Tool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    def forward(self, question):
        user_input = input(f"{question} => Type your answer here:")
        return user_input

class WikipediaRetrieverTool(Tool):
    name = "web_search"
    description = "Uses semantic search to retrieve the parts of 2018 wikipedia that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, max_results=3, **kwargs):
        super().__init__()
        self.max_results = max_results
        if "port" in kwargs.keys():
            self.port = kwargs["port"]
        else:
            self.port = "8005"
        self.url = f"http://127.0.0.1:{self.port}/retrieve"

    def forward(self, query: str) -> str:
        import requests

        assert isinstance(query, str), "Your search query must be a string"
        payload = {
            "queries": [query],
            "topk": self.max_results,
            "return_scores": True
        }

        # Send POST request
        response = requests.post(self.url, json=payload, timeout=120)

        # Raise an exception if the request failed
        response.raise_for_status()

        # Get the JSON response
        retrieved_data = response.json()
        docs = retrieved_data["result"][0]

        return "\nRetrieved documents:" + "".join(
            [
                f"\n\n===== Document {str(i)}, similarity: {doc['score']:.2f} =====\n" + doc["document"]["contents"]
                for i, doc in enumerate(docs)
            ]
        )

TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        PythonInterpreterTool,
    ]
}

__all__ = [
    "PythonInterpreterTool",
    "FinalAnswerTool",
    "UserInputTool",
    "WikipediaRetrieverTool"
]
