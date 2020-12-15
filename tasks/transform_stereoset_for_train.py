import json
import csv
import re
from pathlib import Path
from datasets import Dataset
import pandas as pd
import spacy

# from allennlp.predictors.predictor import Predictor
# import allennlp_models.structured_prediction
# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

nlp = spacy.load("en_core_web_lg")
p = inflect.engine()

DATASET_PATH = Path("data/stereoset_dev.json")
OUTPUT_PATH = Path("data") / 'stereoset_train.csv'

RELIGION_PEOPLE_SET = {"Muslim, Brahmin", "Christian", "Buddhist", "Jew", "Atheist", "Catholic", "Protestant"}
RELIGION_ITEM_SET = {"Bible", "Koran", "Sutra"}
RACE_SET = {'Ethiopian', 'Cameroon', 'Sierra Leon', 'Iranian', 'Lebanon', 'Spain', 'Eriteria', 'Morocco', 'Crimean', 'Japanese', 'Iraq', 'Columbian', 'Yemen', 'African', 'Britain', 'Norweigan', 'Russian', 'Bengali', 'Syria', 'Jordan', 'Hispanic', 'Ghanaian', 'Persian people', 'Vietnam', 'Arab', 'Eritrean', 'Italy', 'Ukrainian', 'Somalia', 'Bangladesh', 'Saudi Arabian', 'Cape Verde', 'Norway', 'Ecuador', 'Afghanistan', 'Ethiopia', 'White', 'Black', 'Asian'}
PROFESSION_SET = {'musician', 'civil servant', 'football player', 'software developer', 'policeman', 'tailor', 'mover', 'prisoner', 'plumber', 'producer', 'nurse', 'engineer', 'performing artist', 'physicist', 'manager', 'politician', 'researcher', 'commander', 'assistant', 'entrepreneur', 'mathematician', 'butcher', 'delivery man', 'chess player', 'bartender', 'prosecutor', 'chemist', 'psychologist', 'guitarist', 'historian'}

SOCIAL_GROUP_SETS = {
    "race": RACE_SET,
    "religion_people": RELIGION_PEOPLE_SET,
    "religion_item": RELIGION_ITEM_SET,
    "profession": PROFESSION_SET
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

    df_stereoset = pd.DataFrame({"stereo_antistereo": [], "bias_type": [], "sent_more": [], "target": []})
    for i, data in enumerate(dataset):
        sentences = data["sentences"]
        if data["bias_type"] == "religion":
            bias_type = "religion_people" if data["target"] in RELIGION_PEOPLE_SET else "religion_item"
        else:
            bias_type = data["bias_type"]

        for sent in sentences:
            df_stereoset = df_stereoset.append({"stereo_antistereo": sent["gold_label"], "bias_type": bias_type, "sent_more": sent["sentence"], "target": data["target"]}, ignore_index=True)

    df_stereoset = df_stereoset[df_stereoset["stereo_antistereo"] != "unrelated"]
    df_output = pd.DataFrame({"stereo_antistereo": [], "bias_type": [], "sent_more": [], "sent_less": []})

    for i, row in df_stereoset.iterrows():
        target = row["target"]
        if row["bias_type"] == "gender":
            sent = row["sent_more"]
            doc = nlp(sent)
            for term1, term2 in GENDER_SUBWORDSWAP_MAP.items():
                sent = swap_term_in_sentence(sent, term1, term2)

            for token in doc:
                if token.text.lower() == "her":
                    if token.tag_ == "PRP":
                        sent = swap_token_in_sent(sent, token, "him")
                    elif token.tag_ == "PRP$":
                        sent = swap_token_in_sent(sent, token, "his")
                else:
                    tkn = token.text.lower() if token.tag_ != "NNS" else token.lemma_.lower()
                    if tkn in GENDER_WORDSWAP_MAP.keys():
                        alternative = GENDER_WORDSWAP_MAP[tkn] if token.tag_ != "NNS" else p.plural(GENDER_WORDSWAP_MAP[tkn])
                        sent = swap_token_in_sent(sent, token, alternative)

            df_output = df_output.append({
                "sent_more": row["sent_more"],
                "sent_less": sent,
                "stereo_antistereo": row["stereo_antistereo"],
                "bias_type": row["bias_type"]
            }, ignore_index=True)
        else:
            groups = SOCIAL_GROUP_SETS[row["bias_type"]]
            other_groups = groups - {row["target"]}
            for group in other_groups:
                sent = row["sent_more"]
                sent = sent.replace(target.lower(), group.lower()).replace(target.capitalize(), group.capitalize())
                df_output = df_output.append({
                    "sent_more": row["sent_more"],
                    "sent_less": sent,
                    "stereo_antistereo": row["stereo_antistereo"],
                    "bias_type": row["bias_type"]
                }, ignore_index=True)

    df_output.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    transoform_dataset()
