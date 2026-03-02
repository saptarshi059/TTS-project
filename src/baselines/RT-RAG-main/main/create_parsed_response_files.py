import json
from argparse import ArgumentParser


def parse_qa_file(file_path):
    """Parse QA file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    qa_blocks = content.strip().split('---\n')
    qa_pairs = []

    for block in qa_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        qa_pair = {}

        for line in lines:
            if line.startswith('qid:'):
                qa_pair['qid'] = line.replace('qid:', '').strip()
            elif line.startswith('question:'):
                qa_pair['question'] = line.replace('question:', '').strip()
            elif line.startswith('predicted_answer:'):
                qa_pair['predicted_answer'] = line.replace('predicted_answer:', '').strip()
            elif line.startswith('golden_answers:'):
                golden_answers_text = line.replace('golden_answers:', '').strip()
                try:
                    qa_pair['golden_answers'] = json.loads(golden_answers_text)
                except json.JSONDecodeError:
                    if ',' in golden_answers_text:
                        qa_pair['golden_answers'] = [item.strip() for item in golden_answers_text.split(',')]
                    else:
                        qa_pair['golden_answers'] = [golden_answers_text]

        if len(qa_pair) == 4:
            qa_pairs.append(qa_pair)

    return qa_pairs

def main(dataset):
    base_path = f"output/results/{dataset}/dense_chunk200_topk1_20_topk2_10"
    response_file = base_path + "/1.txt"
    parsed_file = base_path + f"/{dataset}_parsed.jsonl"

    qa_pairs = parse_qa_file(response_file)
    with open(parsed_file, 'w') as outfile:
        for entry in qa_pairs:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(dataset=args.dataset)