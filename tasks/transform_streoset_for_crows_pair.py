from pathlib import Path
from datasets import Dataset
import pandas as pd
import json
import csv

DATASET_PATH = Path("data/stereoset_dev.json")

def transoform_dataset():

    with open(DATASET_PATH, 'r') as f:
        stereoset_dict = json.load(f)

    stereoset = pd.json_normalize(stereoset_dict['data']['intersentence'])
    dataset = Dataset.from_pandas(stereoset)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    transoform_dataset()
