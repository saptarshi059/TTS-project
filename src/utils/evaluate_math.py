from sympy.parsing.latex import parse_latex
from argparse import ArgumentParser
from datasets import load_dataset
import pandas as pd
import sympy as sp
import statistics
import re

def extract_all_answers(text):
    """Extract all boxed answers from text."""
    text = text.replace('\\\\', '\\')
    results = []

    for match in re.finditer(r'\\?boxed\{', text):
        start = match.end()
        depth = 1
        for i, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    results.append(text[start:start + i].strip())
                    break

    return results

def extract_answer(text):
    """Return last boxed answer, with fallback."""
    answers = extract_all_answers(text)
    if answers:
        return answers[-1]

    # Fallback: last "the answer is X" or "= X"
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


def try_parse_latex(expr):
    try:
        return parse_latex(expr)
    except Exception:
        return None

def strip_outer(s):
    s = s.strip()
    if (s.startswith('(') and s.endswith(')')) or \
       (s.startswith('[') and s.endswith(']')):
        return s[1:-1]
    # Mixed interval brackets: (-\infty, 0] or [0, \infty)
    if (s.startswith('(') and s.endswith(']')) or \
       (s.startswith('[') and s.endswith(')')):
        return s[1:-1]
    return s


def is_interval(s):
    """Check if expression is an interval like (-\infty, 0] or [1, \infty)."""
    return bool(re.match(r'^[\(\[]-?\\?infty|^[\(\[].*,.*[\)\]]$', s))


def parse_interval(s):
    """Convert interval string to sympy Interval."""
    s = s.strip()
    left_open = s[0] == '('
    right_open = s[-1] == ')'
    inner = s[1:-1]
    parts = inner.split(',')
    if len(parts) != 2:
        return None
    try:
        left = parse_latex(parts[0].strip().replace('\\infty', 'oo').replace('infty', 'oo'))
        right = parse_latex(parts[1].strip().replace('\\infty', 'oo').replace('infty', 'oo'))
        return sp.Interval(left, right, left_open=left_open, right_open=right_open)
    except Exception:
        return None


def sympy_equivalent(gold, pred):
    gold = strip_outer(gold)
    pred = strip_outer(pred)

    # Interval comparison: (-\infty, 0] style
    if is_interval(gold) or is_interval(pred):
        g_interval = parse_interval(gold) if is_interval(gold) else None
        p_interval = parse_interval(pred) if is_interval(pred) else None
        if g_interval is None or p_interval is None:
            return False
        return g_interval == p_interval

    # Tuple/coordinate: compare element-wise
    if ',' in gold and ',' in pred:
        gold_parts = gold.split(',')
        pred_parts = pred.split(',')
        if len(gold_parts) != len(pred_parts):
            return False
        try:
            return all(
                sp.simplify(parse_latex(g.strip()) - parse_latex(p.strip())) == 0
                for g, p in zip(gold_parts, pred_parts)
            )
        except Exception:
            return False

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
    if gold_is_extracted:
        gold = gold_raw.replace('\\\\', '\\').strip()
    else:
        gold = extract_answer(gold_raw)

    if gold is None:
        return False

    # Check all model answers — True if any match
    pred_candidates = extract_all_answers(model_raw)
    if not pred_candidates:
        # Fall back to the fallback extractor
        pred = extract_answer(model_raw)
        pred_candidates = [pred] if pred else []

    if not pred_candidates:
        return False

    for pred in pred_candidates:
        if normalize(gold) == normalize(pred):
            return True
        if sympy_equivalent(normalize(gold), normalize(pred)):
            return True

    return False


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