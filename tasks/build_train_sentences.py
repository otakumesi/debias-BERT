import json

from allennlp_models.common.ontonotes import Ontonotes

DATASET_PATH = 'data/ontonotes_conll'

PARENS = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LSB-': '[',
    '-RSB-': ']',
    '-LCB-': '{',
    '-RCB-': '}'
}

def build_swap_dict():
    swap_dict = {}
    for line in open('data/generalized_swaps.txt').readlines():
        word_pair = line.split('\t')
        l_word = word_pair[0].strip()
        r_word = word_pair[1].strip()
        swap_dict[l_word] = r_word
        swap_dict[r_word] = l_word
    return swap_dict

SWAP_DICT = build_swap_dict()

def read_ontonotes(file_path: str):
    reader = Ontonotes()
    for sentence in reader.dataset_iterator(file_path):
        yield sentence

def build_sentence_dicts(dataset_path: str):
    sent_dicts = []
    swap_keys = list(SWAP_DICT.keys())
    for sentence in read_ontonotes(dataset_path):
        orig_words = [PARENS.get(word, word.lower()) for word in sentence.words]

        if not any([w in swap_keys for w in orig_words]):
            continue

        opos_words = [SWAP_DICT.get(word, word) for word in orig_words]

        sent_dict = {'orig_sentence': ' '.join(orig_words),
                     'swapped_sentence': ' '.join(opos_words) }
        sent_dicts.append(sent_dict)

    return json.dumps({'data': sent_dicts})

if __name__ == '__main__':
    print(build_sentence_dicts(DATASET_PATH))
