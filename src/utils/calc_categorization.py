import pandas as pd

s = pd.read_json('streamed_responses.jsonl', lines=True)
c, p, w, o = 0, 0, 0, 0

for row in s.itertuples():
    op = row.generation.split('PREDICTION: ')[-1]
    if op == 'Correct':
        c += 1
    elif op == 'Partial':
        p += 1
    elif op == 'Wrong':
        w += 1
    else:
        o += 1

print(f"Correct: {c} | Partial: {p} | Wrong: {w} | Other: {o}")
