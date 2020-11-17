from pathlib import Path
import re
import csv

import spacy

from datasets import load_dataset

MASK_NAMES_PATH = Path('data') / 'masking_names.txt'
CROWS_DATASET_PATH = Path("data") / "crows_pairs_anonymized.csv"
MASKED_CROWS_DATASET_PATH = Path("data") / "masked_crows_pairs_anonymized.csv"

nlp = spacy.load("en_core_web_lg")


def transform():
    dataset = load_dataset("csv", data_files=str(CROWS_DATASET_PATH), split="train")

    with open(MASK_NAMES_PATH, 'r') as f:
       names = [line.strip() for line in f.readlines()]

    def mask_sent_names(example):
        sent_more = example['sent_more']
        sent_less = example['sent_less']

        for name in names:
            if name in sent_more:
                sent_more = re.sub(f"{name}(\s|'s)", '[MASK]\\1', sent_more)
            if name in sent_less:
                sent_less = re.sub(f"{name}(\s|'s)", '[MASK]\\1', sent_less)

        return {'sent_more': sent_more, 'sent_less': sent_less}


    dataset = dataset.map(mask_sent_names)

    with open(MASKED_CROWS_DATASET_PATH, 'w') as f:
        writer = csv.writer(f)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    transform()
