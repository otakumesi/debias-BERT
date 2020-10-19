from tqdm import tqdm
import re
from datasets import load_dataset

WIKI_SENTENCES = 'data/wiki.sentences.txt'

dataset = load_dataset("wiki40b", 'en', split='train')

with open(WIKI_SENTENCES, 'w') as f:
    for doc in dataset['text']:
        paragraphs = re.findall(r'_START_PARAGRAPH_\n(.+)\n?', doc)
        paragraphs = sum([para.split('_NEWLINE_') for para in paragraphs], [])
        paragraphs = [para.lower() for para in paragraphs]
        for paragraph in paragraphs:
            sents = paragraph.split('_NEWLINE_')
            for sent in sents:
                f.write(f'{sent}\n')
