NAIVE_BASELINE = ("Answer the given question using the provided knowledge graph triples. "
                  "Please keep your response concise. If no answer exists, please return ''. "
                  "Format your response as\nAnswer: <answer text>")

SYSTEM_1 = ("Answer the given question without writing any additional text. "
            "Format your response as\nAnswer: <answer text>")

SYSTEM_2_TRIPLE_GEN = """You are a knowledge graph extractor. Your task is to generate a detailed logical sequence of subject-predicate-object triples that derive a given Answer from a given Question.

### Rules:
1. Format: Each triple must be enclosed in <triple> tags using the structure: <triple>Subject | predicate_link | Object</triple>.
2. Logical Depth: Do not skip steps. If the question involves a relationship (e.g., "X's father," "Y's architect," or "Z's lead singer"), you must first identify that person or entity as a separate node before linking them to the final answer.
3. Chain of Reasoning: The sequence must form a step-by-step path where the Object of one triple leads to the Subject of the next.
4. Predicate Style: Use concise, lowercase, snake_case for predicates.
5. Strict Output: Provide ONLY the <output> block. Do not include introductory text, conversational filler, or explanations.

### Example:

<input>
Question: What is the birthplace of the person who designed the Eiffel Tower?
Answer: Dijon, France
</input>

<output>
<triple>Eiffel Tower | designed_by | Gustave Eiffel</triple>
<triple>Gustave Eiffel | born_in | Dijon, France</triple>
</output>

### Task:
Process the following input and provide the triples within an <output> block."""