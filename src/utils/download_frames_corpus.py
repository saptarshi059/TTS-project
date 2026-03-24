from urllib.parse import unquote
from tqdm import tqdm
import pandas as pd
import wikipediaapi
import time
import ast

def main():
    frames_dataset = pd.read_json("../../sampled_data/frames/sampled_ds.json")

    wiki = wikipediaapi.Wikipedia(
        user_agent="FramesBot/1.0 (contact: your@email.com)",
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    all_links = []
    for row in frames_dataset.itertuples():
        all_links.extend(ast.literal_eval(row.wiki_links))

    all_texts = []
    print("Downloading pages...")
    for url in tqdm(all_links):
        page_name = unquote(url.split('/')[-1])
        time.sleep(0.1)
        page = wiki.page(page_name)
        if not page.exists():
            continue

        all_texts.append(page.text)

    print("Saving corpus...")
    pd.DataFrame(columns=["URL", "Text"], data=zip(all_links, all_texts)).to_parquet("../../all_data/frames/frames_corpus")


if __name__ == "__main__":
    main()