from pathlib import Path
from datasets import Dataset
import pandas as pd
import json
import csv

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

DATASET_PATH = Path("data/stereoset_dev.json")
OUTPUT_PATH = Path("data") / 'stereoset_train.csv'

RELIGION_PEOPLE_SET = {"Muslim, Brahmin", "Christian", "Buddhist", "Jew", "Atheist"}
RELIGION_ITEM_SET = {"Bible", "Koran", "Sutra"}
RACE_SET = {'Ethiopian', 'Cameroon', 'Sierra Leon', 'Iranian', 'Lebanon', 'Spain', 'Eriteria', 'Morocco', 'Crimean', 'Japanese', 'Iraq', 'Columbian', 'Yemen', 'African', 'Britain', 'Norweigan', 'Russian', 'Bengali', 'Syria', 'Jordan', 'Hispanic', 'Ghanaian', 'Persian people', 'Vietnam', 'Arab', 'Eritrean', 'Italy', 'Ukrainian', 'Somalia', 'Bangladesh', 'Saudi Arabian', 'Cape Verde', 'Norway', 'Ecuador', 'Afghanistan', 'Ethiopia', 'White', 'Black', 'Asian'}
PROFESSION_SET = {'musician', 'civil servant', 'football player', 'software developer', 'policeman', 'tailor', 'mover', 'prisoner', 'plumber', 'producer', 'nurse', 'engineer', 'performing artist', 'physicist', 'manager', 'politician', 'researcher', 'commander', 'assistant', 'entrepreneur', 'mathematician', 'butcher', 'delivery man', 'chess player', 'bartender', 'prosecutor', 'chemist', 'psychologist', 'guitarist', 'historian'}

SOCIAL_GROUP_SETS = {
    "race": RACE_SET,
    "religion_people": RELIGION_PEOPLE_SET,
    "religion_item": RELIGION_ITEM_SET,
    "profession": PROFESSION_SET
}

GENDER_SWAP_MAP = {
    "she": "he",
    "woman": "man",
    "women": "men",
    "mother": "father",
    "female": "male",
    "herself": "himself",
    "mommy": "daddy",
    "girl": "boy",
    "sister": "brother"
}


# spaceを考慮すべき（他の単語が変換されるので）
def swap_term_in_sentence(sent, term1, term2):
    sent = sent.replace(term1.lower(), "[SWAP_TEMP_LOWER]").replace(term1.capitalize(), "[SWAP_TEMP_CAPIT]")
    sent = sent.replace(term2.lower(), term1.lower()).replace(term2.capitalize(), term1.capitalize())
    sent = sent.replace("[SWAP_TEMP_LOWER]", term2.lower()).replace("[SWAP_TEMP_CAPIT]", term2.capitalize())
    return sent


def transoform_dataset():

    with open(DATASET_PATH, 'r') as f:
        stereoset_dict = json.load(f)

    stereoset = pd.json_normalize(stereoset_dict['data']['intrasentence'])
    dataset = Dataset.from_pandas(stereoset)

    df_stereoset = pd.DataFrame({"stereo_antistereo": [], "bias_type": [], "sent_more": [], "sent_less": [], "target": []})
    for i, data in enumerate(dataset):
        sentences = data["sentences"]
        if data["bias_type"] == "religion":
            bias_type = "religion_people" if data["target"] in RELIGION_PEOPLE_SET else "religion_item"
        else:
            bias_type = data["bias_type"]

        for sent in sentences:
            df_stereoset = df_stereoset.append({"stereo_antistereo": sent["gold_label"], "bias_type": bias_type, "sent_more": sent["sentence"], "target": data["target"]}, ignore_index=True)

    df_stereoset = df_stereoset[df_stereoset["stereo_antistereo"] != "unrelated"]

    for i, row in df_stereoset.iterrows():
        target = row["target"]
        if row["bias_type"] == "gender":
            sent = row["sent_more"]
            import ipdb; ipdb.set_trace()
            predictor.predict(sentence=sent)

            for term1, term2 in GENDER_SWAP_MAP.items():
                sent = swap_term_in_sentence(sent, term1, term2)
            sent = sent.replace(target.lower(), another_term.lower()).replace(target.capitalize(), another_term.capitalize())
            df_stereoset["sent_less"] = sent
        else:
            groups = SOCIAL_GROUP_SETS[row["bias_type"]]
            other_groups = groups - {row["target"]}
            for group in other_groups:
                sent = row["sent_more"]
                sent = sent.replace(target.lower(), group.lower()).replace(target.capitalize(), group.capitalize())
                df_stereoset["sent_less"] = sent

    df_stereoset[["sent_more", "sent_less", "bias_type", "stereo_antistereo"]].to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    transoform_dataset()
