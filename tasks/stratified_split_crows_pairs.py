from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_FOLDER = Path("data")
CROWS_PAIRS_PATH = DATASET_FOLDER / "crows_pairs_anonymized.csv"


def main():
    df = pd.read_csv(CROWS_PAIRS_PATH)

    columns = ["sent_more", "sent_less", "stereo_antistereo", "bias_type", "annotations", "anon_writer", "anon_annotators"]
    df_train, df_valid_test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42, shuffle=True, stratify=df["bias_type"])
    print(f"--- output train_set... {len(df_train)} data ---")
    df_train.to_csv(DATASET_FOLDER / "crows_pairs_train.csv")

    df_valid, df_test = train_test_split(df_valid_test, train_size=0.75, test_size=0.25, stratify=df_valid_test["bias_type"])
    print(f"--- output valid_set... {len(df_valid)} data ---")
    df_valid.to_csv(DATASET_FOLDER / "crows_pairs_valid.csv")

    print(f"--- output test_set... {len(df_test)} data ---")
    df_test.to_csv(DATASET_FOLDER / "crows_pairs_test.csv")


if __name__ == '__main__':
    main()
