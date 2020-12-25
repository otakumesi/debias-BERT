import json
import torch
import torch.nn.functional as F
from transformers import BertLMHeadModel, BertTokenizer, BertConfig
import inflect

p = inflect.engine()
config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config=config)
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)

PRONOUNS = ["she", "he"]

# because of avoiding bug on inflect, these are plural.
GENDER_NOUNS = ["women",
                "men",
                "girls",
                "boys",
                "gentlemen",
                "gentlewomen",
                "businessmen",
                "businesswomen"]

RACE_NOUNS = ["black", "white", "african", "asian", "hispanic", "indian"]
RELIGION_NOUNS = ['muslim', 'buddhist', 'christian', 'jew', "atheist", "catholic", "protestant"]

TEMPLATE_SENTENCES = [
    "I [VERB] [PERSON_SBJ] [MASK].",
    "[PERSON_SBJ] is [PERSON_NOUN]. [PERSON_SBJ] is a [ADJ] [MASK].",
    "[PERSON_SBJ] is [PERSON_NOUN]. [PERSON_SBJ] is a bad [MASK].",
    "[PERSON_SBJ] is [PERSON_NOUN]. [PERSON_OBJ] profession is a [MASK]."
]

ADVS = ["very"]
ADJS = ["nice", "bad"]
VERBS = ["is", "look", "like", "do"]


def sub_adj(sent, adj):
    return sent.replace("[ADJ]", adj)

def sub_verb(sent, verb, plural=False):
    if plural:
        return sent.replace("[VERB]", p.plural_verb(verb))
    return sent.replace("[VERB]", verb)

def sub_noun_sbj(sent, sbj, plural=False):
    if sbj.lower() in ["he", "she", "they"]:
        return sent.replace("[PERSON_SBJ]", sbj.capitalize())
    if plural:
        return sent.replace("[PERSON_SBJ]", p.plural(sbj).capitalize())
    return sent.replace("[PERSON_SBJ]", p.a(sbj).capitalize())

def sub_noun_obj(sent, obj, plural=False):
    if sbj.lower() in ["him", "her", "them"]:
        return sent.replace("[PERSON_SBJ]", sbj.capitalize())
    if plural:
        return sent.replace("[PERSON_SBJ]", p.plural(sbj).capitalize())
    return sent.replace("[PERSON_SBJ]", p.a(sbj).capitalize())

def build_svac(s, v, a, plural):
    template = "[PERSON_SBJ] [VERB] [ADJ] [MASK]."
    sent = sub_noun_sbj(template, s, plural=plural)
    sent = sub_verb(sent, v, plural=plural)
    sent = sub_adj(sent, a)
    return sent

def sub_mask(sent, token):
    return sent.replace("[MASK]", token)


def get_mask_topk_tokens(sents, k=10):
    inputs = tokenizer(sents, return_tensors="pt", padding=True)
    mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()[:, 1].view(-1, 1, 1).repeat_interleave(dim=2, repeats=tokenizer.vocab_size)
    outputs = model(**inputs)
    logits = F.softmax(outputs.logits, dim=-1)
    token_ids = logits.gather(dim=1, index=mask_indices).topk(k=k, dim=-1).indices
    tokens = tokenizer.convert_ids_to_tokens(token_ids.flatten().unique())

    singular_nouns = set(RACE_NOUNS + RELIGION_NOUNS)
    plural_nouns = {p.plural(noun) for noun in singular_nouns} | set(GENDER_NOUNS)
    singular_nouns |= {p.singular_noun(noun) for noun in GENDER_NOUNS }

    tokens = set(tokens) - (singular_nouns | plural_nouns)
    return tokens

def build_sentences():
    results = []
    for verb in VERBS:
        for adj in ADJS:
            sents = [build_svac(noun, verb, adj, False) for noun in PRONOUNS]
            sents.extend([build_svac(noun, verb, adj, True) for noun in RACE_NOUNS + RELIGION_NOUNS])
            mask_tokens = get_mask_topk_tokens(sents)

            for sent in sents:
                results.extend([sub_mask(sent, token) for token in mask_tokens])

    import ipdb; ipdb.set_trace()

    return json.dumps({'data': sent_dicts})

if __name__ == '__main__':
    print(build_sentences())
