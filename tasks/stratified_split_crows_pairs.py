from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_FOLDER = Path("data")
CROWS_PAIRS_PATH = DATASET_FOLDER / "crows_pairs_anonymized.csv"


def main():
    df = pd.read_csv(CROWS_PAIRS_PATH)
    sss = (n_splits=1, train_size=0.7, test_size=0.3, random_state=42)

    columns = ["sent_more", "sent_less", "stereo_antistereo", "bias_type", "annotations", "anon_writer", "anon_annotators"]
    for indeces_train, indeces_test in train_test_split(df, train_size=0.7, test_size=0.3, random_state=42, shuffle=True, df["bias_type"]):
        df[columns].iloc[indeces_train].to_csv(DATASET_FOLDER / "crows_pairs_train.csv")
        df[columns].iloc[indeces_test].to_csv(DATASET_FOLDER / "crows_pairs_test.csv")


if __name__ == '__main__':
    main()
