import json
import csv
import re
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import spacy
import inflect


nlp = spacy.load("en_core_web_lg")
p = inflect.engine()

DATASET_PATH = Path("data/stereoset_dev.json")
OUTPUT_PATH = Path("data") / 'stereoset_train.csv'

RELIGION_PEOPLE_SET = {"Muslim", "Brahmin"}
RELIGION_ITEM_SET = {"Bible"}
RACE_NATION_SET = {"Cameroon", "Sierra Leon", "Lebanon", "Spain", "Britain", "Eriteria", "Morocco", "Crimean", "Iraq", "Yemen", "Syria", "Jordan", "Vietnam", "Italy", "Somalia", "Bangladesh", "Cape Verde", "Norway", "Ecuador", "Afghanistan", "Ethiopia"}
RACE_ETHINIC_SET = {"Ethiopian", "Iranian", "Japanese", "Columbian", "Norweigan", "Russian", "Bengali", "Ghanaian", "Persian people", "Arab", "Eritrean", "Ukrainian", "Saudi Arabian"}
RACE_INNER_ETHNIC_SET = {"Hispanic", "African"}

PROFESSION_SET = {"musician", "civil servant", "football player", "software developer", "policeman", "tailor", "mover", "prisoner", "plumber", "producer", "nurse", "engineer", "performing artist", "physicist", "manager", "politician", "researcher", "commander", "assistant", "entrepreneur", "mathematician", "butcher", "delivery man", "chess player", "bartender", "prosecutor", "chemist", "psychologist", "guitarist", "historian"}

SOCIAL_GROUP_SETS = {
    "religion_people": RELIGION_PEOPLE_SET,
    "religion_item": RELIGION_ITEM_SET,
    "race_nation": RACE_NATION_SET,
    "race_ethnic": RACE_ETHINIC_SET,
    "race_inner": RACE_INNER_ETHNIC_SET,
    "profession": PROFESSION_SET
}

AUGMENT_MAPS = {
    "Ecuador": {"Mexico"},
    "Muslim": {"Atheist"},
    "Japanese": {"Chinese", "Asian"},
    "African": {"Black", "Native American"},
    "Hispanic": {"Latino"}
}

ALTERNATIVE_GROUP_SETS = {
    "religion_people": {"Christian", "Catholic", "Protestant", "Buddhist"},
    "religion_item": {"Koran", "Sutra"},
    "race_nation": {"United States", "UK", "USA"},
    "race_ethnic": {"American", "English"},
    "race_inner": {"White"}
}

GENDER_WORDSWAP_MAP = {
    "he": "she",
    "she": "he",
    "mother": "father",
    "father": "mother",
    "female": "male",
    "herself": "himself",
    "himself": "herself",
    "mommy": "daddy",
    "daddy": "mommy",
    "sister": "brother",
    "brother": "sister",
    "daughter": "son",
    "son": "daughter",
    "hasband": "wife",
    "wife": "hasband",
    "ms.": "mr.",
    "mr.": "mr."
}

GENDER_SUBWORDSWAP_MAP = {
    "woman": "man",
    "women": "men",
    "girl": "boy"
}


def swap_term_in_sentence(sent, term1, term2):
    sent = re.sub(rf"({term1.lower()})(s|\s|\.|\?|\!|')", r"%SWAP_TEMP_LOWER%\2", sent)
    sent = re.sub(rf"({term1.capitalize()})(s|\s|\.|\?|\!|')", r"%SWAP_TEMP_CAPIT%\2", sent)

    sent = re.sub(rf"({term2.lower()})(s|\s|\.|\?|\!|')", rf"{term1.lower()}\2", sent)
    sent = re.sub(rf"({term2.capitalize()})(s|\s|\.|\?|\!|')", rf"{term1.capitalize()}\2", sent)

    sent = sent.replace("%SWAP_TEMP_LOWER%", term2.lower()).replace("%SWAP_TEMP_CAPIT%", term2.capitalize())

    return sent


def swap_token_in_sent(sent, target_token, alternative):
    token_idx = target_token.idx - 1 if target_token.idx != 0 else 0

    if target_token.is_lower:
        sent = sent[:token_idx] + sent[token_idx:].replace(target_token.text, alternative.lower(), 1)
    else:
        sent = sent[:token_idx] + sent[token_idx:].replace(target_token.text, alternative.capitalize(), 1)
    return sent


def transoform_dataset():

    with open(DATASET_PATH, 'r') as f:
        stereoset_dict = json.load(f)

    stereoset = pd.json_normalize(stereoset_dict['data']['intrasentence'])
    dataset = Dataset.from_pandas(stereoset)

    df_stereoset = pd.DataFrame({"stereo_antistereo": [], "bias_type": [], "sent_more": [], "sent_less": [], "target": []})
    for i, data in enumerate(tqdm(dataset)):
        sentences = data["sentences"]
        bias_type = data["bias_type"]

        if bias_type == "religion":
            bias_type = "religion_people" if data["target"] in RELIGION_PEOPLE_SET else "religion_item"

        if bias_type == "race":
            if data["target"] in RACE_NATION_SET:
                bias_type = "race_nation"
            elif data["target"] in RACE_ETHINIC_SET:
                bias_type = "race_ethnic"
            else:
                bias_type = "race_inner"

        for sent in sentences:
            if sent["gold_label"] == "unrelated":
                continue

            if bias_type == "gender":
                sent_less = sent["sentence"]

                for term1, term2 in GENDER_SUBWORDSWAP_MAP.items():
                    sent_less = swap_term_in_sentence(sent_less, term1, term2)

                doc = nlp(sent["sentence"])

                for token in doc:
                    if token.text.lower() == "her":
                        if token.tag_ == "PRP":
                            sent_less = swap_token_in_sent(sent_less, token, "him")
                        elif token.tag_ == "PRP$":
                            sent_less = swap_token_in_sent(sent_less, token, "his")
                    else:
                        tkn = token.text.lower() if token.tag_ != "NNS" else token.lemma_.lower()
                        if tkn in GENDER_WORDSWAP_MAP.keys():
                            alternative = GENDER_WORDSWAP_MAP[tkn] if token.tag_ != "NNS" else p.plural(GENDER_WORDSWAP_MAP[tkn])
                            sent_less = swap_token_in_sent(sent_less, token, alternative)

                df_stereoset = df_stereoset.append({
                    "stereo_antistereo": sent["gold_label"],
                    "bias_type": bias_type,
                    "sent_more": sent["sentence"],
                    "sent_less": sent_less,
                    "target": data["target"]
                }, ignore_index=True)

            else:
                groups = SOCIAL_GROUP_SETS[bias_type]
                target = data["target"]
                alternative_groups = ALTERNATIVE_GROUP_SETS[bias_type] if bias_type != "profession" else SOCIAL_GROUP_SETS[bias_type]
                for group in alternative_groups:
                    if group == target:
                        continue
                    sent_less = sent["sentence"].replace(target.lower(), group.lower()).replace(target.capitalize(), group.capitalize())
                    df_stereoset = df_stereoset.append({
                        "stereo_antistereo": sent["gold_label"],
                        "bias_type": bias_type,
                        "sent_more": sent["sentence"],
                        "sent_less": sent_less,
                        "target": target
                    }, ignore_index=True)

                    additional_words = {}
                    if target in AUGMENT_MAPS.keys():
                        additional_words = AUGMENT_MAPS[target]

                    for aug_word in additional_words:
                        sent_aug = sent["sentence"].replace(target.lower(), aug_word.lower()).replace(target.capitalize(), aug_word.capitalize())

                        df_stereoset = df_stereoset.append({
                            "stereo_antistereo": sent["gold_label"],
                            "bias_type": bias_type,
                            "sent_more": sent_aug,
                            "sent_less": sent_less,
                            "target": aug_word
                        }, ignore_index=True)

    df_stereoset.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    transoform_dataset()
