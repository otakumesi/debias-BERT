import json
from pathlib import Path
from collections import defaultdict
from operator import itemgetter
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from datasets import load_dataset

from constants import GENDER_PAIRS, RACE_SETS, RELIGION_SETS, NATIONALITY_SETS

CROWS_DATASET_PATH = Path("data") / "crows_pairs_anonymized.csv"
OUTPUT_PATH = Path("data") / "debias_target_words.csv"


def main():
    dataset = load_dataset("csv", data_files=str(CROWS_DATASET_PATH), split="train")

    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")

    tokens_dict = defaultdict(list)

    for ex in dataset:
        if ex["bias_type"] in ["socioeconomic", "disability", "physical-appearance", "sexual-orientation", "age"]:
            continue
        for sent in [ex["sent_more"], ex["sent_less"]]:
            result = predictor.predict(sent)
            for i, tag in enumerate(result["tags"]):
                if any([tag.endswith(tag_name) for tag_name in ["ORG", "PER", "LOC"]]):
                    tokens_dict[ex["bias_type"]].append(result["words"][i].lower())

    for name, tokens in tokens_dict.items():
        tokens_dict[name] = list(set(tokens))

    tokens_dict["gender"].extend(set(sum(GENDER_PAIRS, [])))
    tokens_dict["race-color"].extend(set(sum(RACE_SETS, [])))
    tokens_dict["religion"].extend(set(sum(RELIGION_SETS, [])))
    tokens_dict["nationality"].extend(set(sum(NATIONALITY_SETS, [])))

    json.dump({"values": [
        [tokens_dict["gender"]],
        [tokens_dict["race-color"]],
        [tokens_dict["religion"]],
        [tokens_dict["nationality"]]
    ]}, open(OUTPUT_PATH, "w"))

if __name__ == "__main__":
    main()
