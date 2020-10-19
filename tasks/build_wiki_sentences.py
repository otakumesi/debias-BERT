from tqdm import tqdm
import requests
from gensim.corpora import WikiCorpus

DUMP_URL = 'https://dumps.wikimedia.org/enwiki/{date}/enwiki-{date}-pages-articles.xml.bz2'
TMP_WIKI_PATH = '/tmp/wikidumps'

res = requests.get(DUMP_URL.format(date = '20201001'), stream=True)
with open(TMP_WIKI_PATH, 'wb') as f:
    for chunk in tqdm(res.iter_content(chunk_size=1024)):
        f.write(chunk)

corpus = WikiCorpus(TMP_WIKI_PATH)

WIKI_SENTENCES = 'data/wiki.sentences.txt'
with open(WIKI_SENTENCES, 'w') as f:
    for sents in tqdm(corpus.get_texts()):
        for sent in sents:
            f.write(f'{sent}\n')
