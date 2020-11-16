from pathlib import Path

import pandas as pd
from datasets import Dataset

DATA_DIR = Path('data') / 'us_names'
CROWS_DATASET_PATH = Path("data") / "crows_pairs_anonymized.csv"


def transform():
   dfs = [pd.read_csv(path, names=['Name', 'Gender', 'Occurrences']) for path in DATA_DIR.glob('yob*.txt')]

   df_names = pd.concat(dfs, axis=0, ignore_index=True).drop('Gender', axis=1).drop('Occurrences', axis=1)
   df_names = df_names[~df_names.duplicated(subset='Name')]

   dataset = load_dataset("csv", data_files=str(CROWS_DATASET_PATH), split="train")

   def replace_name_to_mask(example):
      sent_more = example['sent_more']
      sent_less = example['sent_less']

      # TODO: masking names


   dataset.map(lambda ex: ex['sentence'])


if __name__ == '__main__':
    transform()
