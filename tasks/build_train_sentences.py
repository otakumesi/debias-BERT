from itertools import combinations
import json

import torch
import torch.nn.functional as F
from transformers import BertLMHeadModel, BertTokenizer, BertConfig
import inflect
import pandas as pd

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

ADVS = ["always", "rarely", "generally", "badly", "nicely"]
VERBS = ["is", "look"]


def sub_adj(sent, adj):
    return sent.replace("[ADJ]", adj)

def sub_verb(sent, verb, plural=False):
    if plural:
        return sent.replace("[VERB]", p.plural_verb(verb))
    return sent.replace("[VERB]", verb)

def sub_noun_pos(sent, pos, plural=False, identity="[PERSON_POS]"):
    if pos.lower() in ["his", "her", "their"]:
        return sent.replace(identity, pos.capitalize())
    return sent.replace(identity, f"{pos}'s".capitalize())

def sub_noun_sbj(sent, sbj, plural=False, identity="[PERSON_SBJ]"):
    if sbj.lower() in ["he", "she", "they"]:
        return sent.replace(identity, sbj.capitalize())
    if plural:
        return sent.replace(identity, p.plural(sbj).capitalize())
    return sent.replace(identity, p.a(sbj).capitalize())

def sub_noun_obj(sent, obj, plural=False, identity="[PERSON_OBJ]"):
    if obj.lower() in ["him", "her", "them"]:
        return sent.replace(identity, obj)
    if plural:
        return sent.replace("[PERSON_OBJ]", p.plural(obj))
    return sent.replace(identity, p.a(obj))

def build_svac(s, v, a, plural):
    template = "[PERSON_SBJ] [VERB] [ADJ] [MASK]."
    sent = sub_noun_sbj(template, s, plural=plural)
    sent = sub_verb(sent, v, plural=plural)
    sent = sub_adj(sent, a)
    return sent

def sbj_pro_to_obj(sbj):
    return {
        "he": "him",
        "she": "her"
    }[sbj.lower()]

def sbj_pro_to_pos(sbj):
    return {
        "he": "his",
        "she": "her"
    }[sbj.lower()]

def build_profession_sents(group, gender_sbj, plural):
    template = "[PERSON_SBJ] [VERB] [PERSON_GROUP]. [PERSON_POS] profession is a [MASK]."
    sent = sub_noun_sbj(template, gender_sbj)
    sent = sub_noun_sbj(sent, group, plural=plural, identity="[PERSON_GROUP]")
    sent = sub_noun_pos(sent, sbj_pro_to_pos(gender_sbj))
    sent = sub_verb(sent, "is")
    return sent

def build_impression_sents(group, gender_sbj, plural):
    template = "[PERSON_SBJ] [VERB] [PERSON_GROUP]. I feel [PERSON_OBJ] [MASK]."
    sent = sub_noun_sbj(template, gender_sbj)
    sent = sub_noun_sbj(sent, group, plural=plural, identity="[PERSON_GROUP]")
    sent = sub_noun_obj(sent, sbj_pro_to_obj(gender_sbj))
    sent = sub_verb(sent, "is")
    return sent

def sub_mask(sent, token):
    return sent.replace("[MASK]", token)


def get_mask_topk_tokens(sents, k=15):
    inputs = tokenizer(sents, return_tensors="pt", padding=True)
    mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()[:, 1].view(-1, 1, 1).repeat_interleave(dim=2, repeats=tokenizer.vocab_size)
    outputs = model(**inputs)
    logits = F.softmax(outputs.logits, dim=-1)
    token_ids = logits.gather(dim=1, index=mask_indices).topk(k=k, dim=-1).indices
    tokens = tokenizer.convert_ids_to_tokens(token_ids.flatten().unique())

    singular_nouns = set(RACE_NOUNS + RELIGION_NOUNS)
    plural_nouns = {p.plural(noun) for noun in singular_nouns} | set(GENDER_NOUNS)
    singular_nouns |= {p.singular_noun(noun) for noun in GENDER_NOUNS}

    tokens = set(tokens) - (singular_nouns | plural_nouns)

    return tokens

def build_sentences(output_file):
    results = []
    for verb in VERBS:
        for adj in ADVS:
            word_sets = [(PRONOUNS, False, "gender"), (["women", "men"], True, "gender"), (["girls", "boys"], True, "gender"), (RACE_NOUNS, True, "race"), (RELIGION_NOUNS, True, "religion")]
            for nouns, plural, bias_type in word_sets:
                sents = [build_svac(noun, verb, adj, plural) for noun in nouns]
                mask_tokens = get_mask_topk_tokens(sents)
                for token in mask_tokens:
                    filled_sents = [sub_mask(sent, token) for sent in sents]
                    for l, r in combinations(filled_sents, 2):
                        results.append((l, r, bias_type, "stereo"))

    for pro in PRONOUNS:
        for nouns, plural, bias_type in [(RACE_NOUNS, True, "race"), (RELIGION_NOUNS, True, "religion")]:
            prof_sents = [build_profession_sents(group, pro, plural) for group in nouns]
            mask_tokens = get_mask_topk_tokens(prof_sents)
            for token in mask_tokens:
                prof_filled_sents = [sub_mask(sent, token) for sent in prof_sents]
                for l, r in combinations(filled_sents, 2):
                    results.append((l, r, bias_type, "stereo"))


            imp_sents = [build_impression_sents(group, pro, plural) for group in nouns]
            mask_tokens = get_mask_topk_tokens(imp_sents)
            for token in mask_tokens:
                imp_filled_sents = [sub_mask(sent, token) for sent in imp_sents]
                for l, r in combinations(filled_sents, 2):
                    results.append((l, r, bias_type, "stereo"))

    df = pd.DataFrame.from_records(results, columns=["more_sent", "less_sent", "bias_type", "stereo_antistereo"])
    df.to_csv(output_file)

if __name__ == "__main__":
    build_sentences("data/train_sentences.csv")
