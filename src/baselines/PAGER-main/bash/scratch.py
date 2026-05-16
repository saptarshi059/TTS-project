for ds in datasets:
    #outline prompt
    base = pd.read_json(f'../../../../sampled_data/{ds}/sampled_ds.json')
    init_page_prompt_template = """You are a Page Planning Expert.
    For a given question, you are tasked with generating a page outline based on the theme of the question.
    The outline you generate will serve as the foundation for the subsequent process, during which the outline will be filled with content to create a complete page that assists the reader in answering the given question.
    Please strictly follow the instructions below.

    Input:  
    - Question: {question}

    Task Steps:  

    1. **Reasoning Analysis**
    - First, analyze the theme of the question and thoroughly understand the knowledge required to answer it, as well as the logical relationships between this knowledge. 
    - Based on the theme of the question, generate a page outline and identify the key content that should be covered in each section. 
    - For each section, propose a suitable and appropriate title, ensuring that each title effectively guides the reader to progressively deepen their understanding of the various aspects of the question. 

    1. **Outline Initialization**  
    - Generate the outline:  
        - Use # [Main Title] for the main title. The main title is a concise and comprehensive abstraction of the page content.
        - Use ## [Section Title] for each section title. The section title is the section heading (do not reveal the final answer).
        - Insert special marker <TO BE FILLED> under all section titles. 
    Make sure the sections of the page are ordered logically, building up the reader's understanding toward answering the question.
    The number of sections on the page should be limited to the scope necessary to answer the question, avoiding any overlap of content between sections.

    2. **Output**  
    - First, you should output the reasoning analysis for initializing the outline. 
    - Then, you should generate a special symbol <OUTLINE>, followed by the generated page outline. Note that the content after <OUTLINE> should only include the final generated page outline, without any comments or explanations.
    """
    all_texts = []
    for row in base.itertuples():
        all_texts.append(init_page_prompt_template.format(question=row.question))

    #page prompt
    sub_question_prompt_template = """You are a professional question-generation expert. 
    First, following the order of the page sections, locate and identify the first unfinished section. An unfinished section is defined as one that contains the special symbol <TO BE FILLED> under the section title.
    Please strictly follow the steps and format below to formulate a precise retrieval question for the first unfinished section of the current page, in order to obtain the necessary information to complete that section.


    Input Parameters:
    - Original question: {question}
    - Current page content: {page}

    Task Steps:

    1. **Parse the page plan:**
        - Read the Page and locate the title and topic of the first section that remains unfilled.

    2. **Align with the original question:**
        - Align the section's topic with the core needs of original question to ensure the retrieval question directly serves the original question.

    3. **Design the retrieval question. The question must:**
        - Be focused: target only the core subtopic of that section;
        - Be clear: use search terms that can be directly used in a search engine or knowledge base;
        - Be concise: include no unnecessary background.

    4. **Output format:**
        - Output only one line containing the English retrieval question, with no additional comments or explanations."""
    ds_page = pd.read_json(f'output_data/new_outline_{ds}.jsonl', lines=True)
    for row in ds_page.itertuples():
        all_texts.append(sub_question_prompt_template.format(question=row.question, page=row.init_page))

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
    ds_page = pd.read_json(f'output_data/new_outline_{ds}_page.jsonl', lines=True)
    for row in ds_page.itertuples():
        if row.subquestion_list == []:
            all_texts.append(gen_prompt_template.format(question=row.question, sub_question='',
                                                        docs_text='', page=row.init_page))
        else:
            for idx, subq in enumerate(row.subquestion_list):
                all_texts.append(gen_prompt_template.format(question=row.question, sub_question=subq,
                                                            docs_text=row.doc_list[idx], page=row.init_page))

    #final_prompt
    prompt = """Page:\n{page}\n
    The User asks a question, and the Assistant solves it.
    The system will provide the Assistant with a page containing information relevant to answering the question. The assistant should answer the question by combining the Page with its internal knowledge.
    When the page provides enough knowledge to answer the question, the assistant should strictly follow the knowledge and writing style from the page. When the page does not provide enough knowledge, the assistant should combine its internal knowledge to answer.
    All answers should be as comprehensive and accurate as possible. The assistant should first think through the reasoning process, then provide the precise and short final answer. 
    The output format for the final answer should be enclosed within <answer></answer> tags. You need to first present the reasoning process, then give the final answer, like: “Reasoning process here\n\n<answer> Only the short final answer here </answer>”.
    \n\nUser:{question}\nAssistant:"""
    ds_final = pd.read_json()