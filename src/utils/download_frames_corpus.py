from datasets import load_dataset, tqdm, Dataset
from urllib.parse import unquote
import wikipediaapi
import time
import ast

def main():
    dataset = load_dataset("google/frames-benchmark")['test']

    wiki = wikipediaapi.Wikipedia(
        user_agent="FramesBot/1.0 (contact: your@email.com)",
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    all_links = []
    for row in dataset:
        all_links.extend(ast.literal_eval(row['wiki_links']))

    corpus = {}
    print("Downloading pages...")
    for idx, url in tqdm(enumerate(all_links)):
        page_name = unquote(url.split('/')[-1])
        time.sleep(0.1)
        page = wiki.page(page_name)
        if not page.exists():
            continue

        page_text = page.text
        corpus[str(idx)] = page_text

    corpus_dataset = Dataset.from_dict({"id": list(corpus.keys()), "text": list(corpus.values())})
    corpus_dataset.save_to_disk("../../data/frames/frames_corpus")

if __name__ == "__main__":
    main()