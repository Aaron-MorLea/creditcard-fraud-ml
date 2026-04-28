# data/prepare_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROC_DIR, exist_ok=True)

def main():
    df = pd.read_csv(RAW_PATH)

    # Features y target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # train / temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # val / test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    joblib.dump((X_train, y_train), os.path.join(PROC_DIR, "train.pkl"))
    joblib.dump((X_val, y_val), os.path.join(PROC_DIR, "val.pkl"))
    joblib.dump((X_test, y_test), os.path.join(PROC_DIR, "test.pkl"))

    print("Splits guardados en data/processed/")

if __name__ == "__main__":
    main()