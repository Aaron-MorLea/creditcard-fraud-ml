# models/train_autoencoder.py
import os
import joblib
import numpy as np

from autoencoder import FraudDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
ART_DIR = os.path.join(BASE_DIR, "models", "artifacts")

def main():
    (X_train, y_train) = joblib.load(os.path.join(PROC_DIR, "train.pkl"))

    # Sólo normales (Class=0)
    mask_normal = (y_train == 0)
    X_train_normal = X_train[mask_normal]

    detector = FraudDetector(input_dim=X_train_normal.shape[1], encoding_dim=8, threshold_percentile=95)
    detector.train(X_train_normal.values)

    detector.save(os.path.join(ART_DIR, "fraud_autoencoder"))
    print("Autoencoder entrenado y guardado.")

if __name__ == "__main__":
    main()