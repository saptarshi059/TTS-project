import re
import pandas as pd
import ast

def get_unique_children_queries(data, unique_queries=None):
  if unique_queries is None:
    unique_queries = set()

  if isinstance(data, dict):
    # If 'children' exists and is a non-empty list
    if 'children' in data and isinstance(data['children'], list):
      for child in data['children']:
        # Extract the query if it exists inside the child node
        if isinstance(child, dict) and 'query' in child:
          unique_queries.add(child['query'])

        # Recursively traverse deeper into the child node
        get_unique_children_queries(child, unique_queries)

  elif isinstance(data, list):
    for item in data:
      get_unique_children_queries(item, unique_queries)

  return len(unique_queries)

all_dicts = []
dataset = ['2wikimultihopqa', 'musique', 'hotpotqa']
for ds in dataset:
    s = pd.read_json(f'outputs/{ds}_outputs.jsonl', lines=True)
    s = s.dropna()
    for row in s.itertuples():
        match = re.search(r'### Inference Tree:\s*(\{.*?\})\s*\n\nPlease answer', row.full_prompt, re.DOTALL)

        if match:
            dict_string = match.group(1).strip()
            all_dicts.append(ast.literal_eval(dict_string))

all_subqueries = []
for d in all_dicts:
    all_subqueries.append(get_unique_children_queries(d))

print(f"Avg. subqueries: {sum(all_subqueries)/1500}")