NAIVE_BASELINE = ("Answer the given question using the provided knowledge graph triples. "
                  "Please keep your response concise. If no answer exists, please return ''. "
                  "Format your response as\nAnswer: <answer text>")

SYSTEM_1 = ("Answer the given question without writing any additional text. "
            "Format your response as\nAnswer: <answer text>")

SYSTEM_2_TRIPLE_GEN = """You are a knowledge graph extractor. Your task is to generate a logical sequence of subject-predicate-object triples that derive a given Answer from a given Question.

### Rules:
1. Format: Each triple must be enclosed in <triple> tags using the structure: <triple>Subject | predicate_link | Object</triple>.
2. Logical Chain: The triples must form a step-by-step path (e.g., the Object of the first triple should lead to the Subject of the next).
3. Predicate Style: Use concise, lowercase, snake_case for predicates (e.g., located_in, part_of, discovered_by).
4. Constraint: The final triple's Object must be the provided Answer.
5. Strict Output: Provide ONLY the <output> block. Do not include introductory text, conversational filler, or explanations.

### Example:

<input>
Question: In which continent is the Eiffel Tower located in?
Answer: Europe
</input>

<output>
<triple>Eiffel Tower | located_in | Paris</triple>
<triple>Paris | city_of | France</triple>
<triple>France | located_in | Europe</triple>
</output>

### Task:
Process the following input and provide the triples within an <output> block."""