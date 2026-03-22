from sympy.parsing.latex import parse_latex
from argparse import ArgumentParser
from datasets import load_dataset
import pandas as pd
import sympy as sp
import statistics
import re


def extract_answer(text):
    text = text.replace('\\\\', '\\')

    # Find the LAST \boxed{} occurrence (model may produce multiple in CoT)
    matches = list(re.finditer(r'\\?boxed\{', text))
    if matches:
        match = matches[-1]  # take the last one
        start = match.end()
        depth = 1
        for i, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:start + i].strip()

    # Fallback: last occurrence of "the answer is X" or "= X"
    match = re.findall(r'(?:the answer is|=)\s*([^\s,\.]+)', text, re.IGNORECASE)
    if match:
        return match[-1].strip()

    return None


def normalize(expr):
    """Normalize LaTeX string before comparison."""
    expr = expr.replace('\\\\', '\\')
    expr = expr.replace(' ', '')
    expr = expr.replace('%', '/100')
    expr = expr.replace('\\left(', '(')
    expr = expr.replace('\\right)', ')')
    expr = expr.replace('\\left[', '[')
    expr = expr.replace('\\right]', ']')
    return expr


def strip_outer(s):
    """Strip outer parentheses or brackets."""
    s = s.strip()
    if (s.startswith('(') and s.endswith(')')) or \
       (s.startswith('[') and s.endswith(']')):
        return s[1:-1]
    return s


def try_parse_latex(expr):
    try:
        return parse_latex(expr)
    except Exception:
        return None


def sympy_equivalent(gold, pred):
    """Compare two normalized LaTeX expressions using SymPy."""
    gold = strip_outer(gold)
    pred = strip_outer(pred)

    # Tuple/coordinate: compare element-wise
    if ',' in gold and ',' in pred:
        gold_parts = gold.split(',')
        pred_parts = pred.split(',')
        if len(gold_parts) != len(pred_parts):
            return False
        return all(
            sp.simplify(parse_latex(g.strip()) - parse_latex(p.strip())) == 0
            for g, p in zip(gold_parts, pred_parts)
        )

    # Single expression
    parsed_gold = try_parse_latex(gold)
    parsed_pred = try_parse_latex(pred)
    if parsed_gold is None or parsed_pred is None:
        return False
    try:
        return sp.simplify(parsed_gold - parsed_pred) == 0
    except Exception:
        return False


def answers_equivalent(gold_raw, model_raw, gold_is_extracted=False):
    """
    Compare gold and model answers.
    Set gold_is_extracted=True if gold is already a clean answer field
    (e.g. from MATH-500's dedicated answer column) rather than a full solution string.
    """
    if gold_is_extracted:
        gold = gold_raw.replace('\\\\', '\\').strip()
    else:
        gold = extract_answer(gold_raw)

    pred = extract_answer(model_raw)

    if gold is None or pred is None:
        return False

    # 1. Exact string match after normalization
    if normalize(gold) == normalize(pred):
        return True

    # 2. SymPy symbolic equivalence
    return sympy_equivalent(normalize(gold), normalize(pred))


def main(dataset:str, op_file: str):
    base_dataset = load_dataset(dataset, split='test').to_pandas()
    output_file = pd.read_json(op_file, lines=True)
    responses = []
    for base_row, response_row in zip(base_dataset.itertuples(), output_file.itertuples()):
        responses.append(answers_equivalent(gold_raw=base_row.answer, model_raw=response_row.generation, gold_is_extracted=True))

    print(f"Accuracy: {statistics.mean(responses)*100:.2f}%")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str, default="../../../all_output/HuggingFaceH4_MATH-500/cot/formatted_generations.jsonl")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500")
    args = parser.parse_args()
    main(dataset=args.dataset, op_file=args.output_file)