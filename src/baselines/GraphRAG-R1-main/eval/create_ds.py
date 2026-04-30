import pandas as pd
import json
import os

datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']

for ds in datasets:
    # 1. Load the data
    input_path = f"../../../../sampled_data/{ds}/sampled_ds.json"
    data = pd.read_json(input_path).to_dict(orient='records')

    # 2. Process the rows
    final_rows = []
    for row in data:
        final_rows.append({
            "question": row['question'],
            "answer": row['answer'],
            "label": row['answer']
        })

    # 3. Create the directory if it doesn't exist
    output_dir = f'../datasets/{ds}'
    os.makedirs(output_dir, exist_ok=True)

    # 4. Save the processed data
    output_file = os.path.join(output_dir, 'Question.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in final_rows:
            # Save each dictionary as a single line
            f.write(json.dumps(entry) + '\n')