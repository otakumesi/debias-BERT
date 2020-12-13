from pathlib import Path
import json
import re


DATASET_PATH = Path("data/winobias")

objective2subjective = {
    "his": "he",
    "him": "he",
    "her": "she"
}

def transform_dataset(sentences):
    results = []
    for i, sent in enumerate(sentences):
        words = re.findall(r"\[(.+?)\]", sent)
        answer = words[0]
        target = words[1].lower()
        sent = sent.replace("[", "").replace("]", "")

        pos = sent.find(answer)

        target = objective2subjective.get(target, target)

        squad_format_record = {
            "id": f"sentence_{i}",
            "title": "Winobias Sentence",
            "context": sent,
            "question": f"What does {target} do?",
            "answers": {
                "answer_start": [pos],
                "text": [answer]
            },
        }
        results.append(squad_format_record)
    return results


def transform_winobias():
    sentence_types = {
        "ground": 1,
        "knowledge": 2,
    }

    for anti_pro in ["anti", "pro"]:
        for name, type_num in sentence_types.items():
            for split in ["dev", "test"]:
                target_file = f"{anti_pro}_stereotyped_type{type_num}.txt.{split}"
                print(f"--- Perform {target_file} ---")
                with open(DATASET_PATH / target_file, "r") as f:
                    sentences = f.readlines()
                sentences = [sent.strip() for sent in sentences]
                transformed_records = transform_dataset(sentences)

                with open(DATASET_PATH / f"{anti_pro}_stereotyped_{name}.{split}.json", "w") as f:
                    json.dump({"version": "1.0", "data": transformed_records}, f)


if __name__ == "__main__":
    transform_winobias()

