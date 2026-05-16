import pandas as pd
from transformers import AutoTokenizer
import numpy as np

datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']

all_texts = []
for ds in datasets:
    ds_page = pd.read_json(f'output_data/new_outline_{ds}_page.jsonl', lines=True)
    gen_prompt_template = """You are a professional page content writer.
        Please strictly follow the instructions below.
        First, following the order of the page sections, locate and identify the first unfinished section. An unfinished section is defined as one that contains the special symbol <TO BE FILLED> under the section title.
        Use the retrieved documents, and your internal knowledge to complete the first unfinished section of the page, and generate the page with that section filled in.
        Please note that you should only fill in one unfinished section, and that section must be the first unfinished section on the page. Do not fill in additional sections or fill across different sections.

        Input:  
        - Original question: {question}
        - Sub-question for retrieval: {sub_question}
        - Retrieved documents: {docs_text}
        - Current page: {page}

        Task Steps:

        1. **Section Completion**  
        - Find the first unfinished section in the page, where the section title contains a special placeholder <TO BE FILLED>.
        - Using information from the retrieved documents and your internal knowledge, write a short paragraph under that section. The paragraph should:  
            - Be tightly related to the original question.  
            - Focus strictly on the topic of that section.  
            - Avoid redundant or irrelevant information.  
            - Remove the <TO BE FILLED> placeholder under that section.  
            - Retain <TO BE FILLED> placeholders for all other unfinishe sections.  
            - If you have filled the last unfinished section of the page, ensure that no <TO BE FILLED> placeholders remain in the page.

        2. **Output format**
           - Only output the entire page with the first unfinished section fully filled in. Do not include any comments, explanations, or isolated section content.
           - Your output must be the entire updated page, including the newly filled content, seamlessly integrated into the original page structure.
           - Do not output just the section you filled in—your output must be the entire page, including all content, both existing and newly added.
        """
    for row in ds_page.itertuples():
        if row.subquestion_list == []:
            continue
            # Despite these questions having no docs, we still have to fire them. Thus, we consider their prompt text.
            #all_texts.append(gen_prompt_template.format(question=row.question, sub_question='',
            #                                            docs_text='', page=row.init_page))
        else:
            for idx, subq in enumerate(row.subquestion_list):
                all_texts.extend(row.doc_list[idx])

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
all_reasoning = tokenizer(all_texts)

average_tokens = np.mean([len(ids) for ids in all_reasoning['input_ids']])

print(f"Average tokens per sequence: {average_tokens}")

