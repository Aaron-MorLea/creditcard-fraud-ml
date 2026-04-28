# models/evaluate_models.py
import os
import json
import joblib
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from autoencoder import FraudDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
ART_DIR = os.path.join(BASE_DIR, "models", "artifacts")
MET_DIR = os.path.join(ART_DIR, "metrics")

os.makedirs(MET_DIR, exist_ok=True)

def eval_xgb(X_test, y_test):
    model = xgb.Booster()
    model.load_model(os.path.join(ART_DIR, "xgb_fraud.json"))
    dtest = xgb.DMatrix(X_test)
    scores = model.predict(dtest)
    return scores

def eval_autoencoder(X_test):
    det = FraudDetector(input_dim=X_test.shape[1], encoding_dim=8)
    det.load(os.path.join(ART_DIR, "fraud_autoencoder"))
    _, fraud_scores = det.predict(X_test.values)
    # Convertimos a prob estilo: score alto = más fraude
    # Normalizamos entre 0 y 1 para compararlo visualmente
    fraud_scores = np.clip(fraud_scores, 0, 3)
    fraud_scores = (fraud_scores - fraud_scores.min()) / (fraud_scores.max() - fraud_scores.min() + 1e-8)
    return fraud_scores

def main():
    (X_test, y_test) = joblib.load(os.path.join(PROC_DIR, "test.pkl"))

    y_scores_xgb = eval_xgb(X_test, y_test)
    y_scores_ae = eval_autoencoder(X_test)

    results = {}

    for name, scores in [("xgb", y_scores_xgb), ("autoencoder", y_scores_ae)]:
        # Threshold 0.5 para prob (XGB) y mismo para AE normalizado
        y_pred = (scores >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        roc_auc = roc_auc_score(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "cm": cm.tolist(),
        }

        # Curvas
        fpr, tpr, _ = roc_curve(y_test, scores)
        prec, rec, _ = precision_recall_curve(y_test, scores)

        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC - {name}")
        plt.legend()
        plt.savefig(os.path.join(MET_DIR, f"roc_{name}.png"))
        plt.close()

        plt.figure()
        plt.plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall - {name}")
        plt.legend()
        plt.savefig(os.path.join(MET_DIR, f"pr_{name}.png"))
        plt.close()

    # Confusion matrices comparadas
    plt.figure(figsize=(8,4))
    for i, name in enumerate(["xgb", "autoencoder"]):
        cm = np.array(results[name]["cm"])
        plt.subplot(1,2,i+1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(MET_DIR, "confusion_matrices.png"))
    plt.close()

    with open(os.path.join(MET_DIR, "eval_report.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluación completada. Métricas guardadas en models/artifacts/metrics/")

if __name__ == "__main__":
    main()