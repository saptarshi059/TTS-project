import re
import sympy as sp
from sympy.parsing.latex import parse_latex


def extract_answer(text):
    """Try \boxed{} first, then 'The answer is X' as fallback."""
    text = text.replace('\\\\', '\\')

    # 1. Try \boxed{}
    match = re.search(r'\\?boxed\{', text)
    if match:
        start = match.end()
        depth = 1
        for i, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:start + i].strip()

    # 2. Fallback: "the answer is X" (last occurrence wins)
    match = re.findall(r'(?:the answer is|=)\s*([^\s,\.]+)', text, re.IGNORECASE)
    if match:
        return match[-1].strip()

    return None


def normalize(expr):
    """Light normalization before string match."""
    expr = expr.replace(' ', '')
    expr = expr.replace('%', '/100')  # avoids the \% SyntaxWarning too
    return expr


def answers_equivalent(gold_raw, model_raw):
    gold = extract_answer(gold_raw)
    pred = extract_answer(model_raw)

    if gold is None or pred is None:
        return False

    # 1. Exact match after normalization
    if normalize(gold) == normalize(pred):
        return True

    # 2. SymPy symbolic equivalence
    try:
        return sp.simplify(parse_latex(gold) - parse_latex(pred)) == 0
    except Exception:
        return False

gold = r'The spinner is guaranteed to land on exactly one of the three regions, so we know that the sum of the probabilities of it landing in each region will be 1. If we let the probability of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, from which we have $x=\\boxed{\\frac{1}{4}}$.'
model = r'\boxed{\frac{1}{4}}}'


print(answers_equivalent(gold, model))