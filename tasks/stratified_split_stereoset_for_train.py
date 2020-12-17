from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import seed

DATASET_FOLDER = Path("data")
STEREOSET_PAIRS_PATH = DATASET_FOLDER / "stereoset_train.csv"


def main():
    seed(seed=42)

    columns = ["idx", "stereo_antistereo", "bias_type", "sent_more", "sent_less", "target"]
    df = pd.read_csv(STEREOSET_PAIRS_PATH, names=columns, skiprows=1)

    df_train, df_valid_test = train_test_split(df, train_size=0.9, test_size=0.1, random_state=42, shuffle=True, stratify=df[["bias_type", "target"]])
    print(f"--- output train_set... {len(df_train)} data ---")
    df_train[columns].to_csv(DATASET_FOLDER / "stereoset_train.train.csv")

    df_valid, df_test = train_test_split(df_valid_test, train_size=0.5, test_size=0.5, stratify=df_valid_test["bias_type", "target"])
    print(f"--- output valid_set... {len(df_valid)} data ---")
    df_valid[columns].to_csv(DATASET_FOLDER / "stereoset_train.valid.csv")

    print(f"--- output test_set... {len(df_test)} data ---")
    df_test[columns].to_csv(DATASET_FOLDER / "stereoset_train.test.csv")


if __name__ == '__main__':
    main()
