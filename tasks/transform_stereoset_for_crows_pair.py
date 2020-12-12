from pathlib import Path
from datasets import Dataset
import pandas as pd
import json
import csv

DATASET_PATH = Path("data/stereoset_dev.json")

def transoform_dataset():

    with open(DATASET_PATH, 'r') as f:
        stereoset_dict = json.load(f)

    stereoset = pd.json_normalize(stereoset_dict['data']['intrasentence'])
    dataset = Dataset.from_pandas(stereoset)


    with open(f'data/stereoset_crows.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'sent_more', 'sent_less', 'stereo_antistereo', 'bias_type'])
        for i, data in enumerate(dataset):
            sentences = data["sentences"]
            more_sent = [sent for sent in sentences if sent["gold_label"] == "stereotype"][0]["sentence"]
            less_sent = [sent for sent in sentences if sent["gold_label"] == "anti-stereotype"][0]["sentence"]
            writer.writerow([str(i), more_sent, less_sent, 'stereo', data["bias_type"]])


if __name__ == "__main__":
    transoform_dataset()
