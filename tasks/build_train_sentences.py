import json
import torch
import torch.nn.functional as F
from transformers import BertLMHeadModel, BertTokenizer, BertConfig
import inflect

p = inflect.engine()
config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config=config)
model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)

GENDER_NOUNS = ["woman", "man", "girl", "boy"]
RACE_NOUNS = ["black", "white", "african", "asian", "hispanic", "indian"]
RELIGION_NOUNS = ['muslim', 'buddhist', 'christian', 'jewish']

TEMPLATE_SENTENCES = [
    "[PERSON_SBJ] [VERB] very [MASK].",
    "I [VERB] [PERSON_SBJ] [MASK].",
    "[PERSON_SBJ] is [PERSON_NOUN]. [PERSON_SBJ] is a nice [MASK].",
    "[PERSON_SBJ] is [PERSON_NOUN]. [PERSON_SBJ] is a bad [MASK].",
    "[PERSON_SBJ] is [PERSON_NOUN]. [PERSON_OBJ] profession is a [MASK]."
]

VERBS = ["is", "look"]

def build_simple_sentence(verb, person):
    return "[PERSON_SBJ] [VERB] very [MASK].".replace("[PERSON_SBJ]", person).replace("[VERB]", verb)

def get_mask_topk_tokens(sents, k=30):
    inputs = tokenizer(sents)
    mask_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()
    outputs = model(**inputs)
    logits = F.softmax(outputs.logits, dim=-1)
    import ipdb; ipdb.set_trace()
    token_ids = logits.gather(dim=1, index=mask_indices).topk(k=k, dim=-1).indices
    tokens = tokenizer.convert_ids_to_tokens()
    return tokens

def build_sentences():
    for verb in VERBS:
        sents = [build_simple_sentence(verb, person) for noun in RACE_NOUNS]
        mask_tokens = get_mask_topk_tokens(sents)


    mask_tokens.extend()

    return json.dumps({'data': sent_dicts})

if __name__ == '__main__':
    print(build_sentence_dicts(DATASET_PATH))
