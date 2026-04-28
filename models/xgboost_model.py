# models/xgboost_model.py
import os
import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
ART_DIR = os.path.join(BASE_DIR, "models", "artifacts")

def main():
    (X_train, y_train) = joblib.load(os.path.join(PROC_DIR, "train.pkl"))
    (X_val, y_val) = joblib.load(os.path.join(PROC_DIR, "val.pkl"))

    # Manejar desbalance con scale_pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {ratio:.2f}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": ratio,
        "nthread": 4,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    # Eval rápida
    y_val_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"[XGB] Val ROC-AUC: {auc:.4f}")

    os.makedirs(ART_DIR, exist_ok=True)
    model.save_model(os.path.join(ART_DIR, "xgb_fraud.json"))
    print("Modelo XGBoost guardado.")

if __name__ == "__main__":
    main()