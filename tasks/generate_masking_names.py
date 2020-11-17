from pathlib import Path

import spacy

from datasets import load_dataset

CROWS_DATASET_PATH = Path("data") / "crows_pairs_anonymized.csv"
OUTPUT_FILE = Path('data') / 'masking_names.txt'
nlp = spacy.load("en_core_web_lg")


def generate():
    dataset = load_dataset("csv", data_files=str(CROWS_DATASET_PATH), split="train")

    target_sents = dataset["sent_more"] + dataset["sent_less"]
    names = []
    for sent in target_sents:
       names.extend([ent.text for ent in nlp(sent) if ent.ent_type_ == 'PERSON'])

    names = set(names)
    with open(OUTPUT_FILE, 'w') as f:
        for name in names:
            f.write(name)
            f.write('\n')


if __name__ == '__main__':
    generate()
