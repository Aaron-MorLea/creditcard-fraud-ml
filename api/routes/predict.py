# api/routes/predict.py
from fastapi import APIRouter, HTTPException
from typing import List
import os
import numpy as np
import xgboost as xgb
import joblib

from api.schemas.transaction import Transaction, PredictionResponse
from models.autoencoder import FraudDetector

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART_DIR = os.path.join(BASE_DIR, "models", "artifacts")

_xgb_model = None
_ae_detector = None
_feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def load_models():
    global _xgb_model, _ae_detector
    if _xgb_model is None:
        _xgb_model = xgb.Booster()
        _xgb_model.load_model(os.path.join(ART_DIR, "xgb_fraud.json"))
    if _ae_detector is None:
        # input_dim = 30 (Time, V1..V28, Amount)
        det = FraudDetector(input_dim=30, encoding_dim=8)
        det.load(os.path.join(ART_DIR, "fraud_autoencoder"))
        _ae_detector = det

@router.post("/predict", response_model=PredictionResponse)
async def predict(tx: Transaction):
    try:
        load_models()
        # Construir vector de features
        values = [
            tx.time,
            tx.V1, tx.V2, tx.V3, tx.V4, tx.V5, tx.V6, tx.V7,
            tx.V8, tx.V9, tx.V10, tx.V11, tx.V12, tx.V13, tx.V14,
            tx.V15, tx.V16, tx.V17, tx.V18, tx.V19, tx.V20, tx.V21,
            tx.V22, tx.V23, tx.V24, tx.V25, tx.V26, tx.V27, tx.V28,
            tx.amount,
        ]
        X = np.array([values])

        dmat = xgb.DMatrix(X)
        score_xgb = float(_xgb_model.predict(dmat)[0])

        _, scores_ae = _ae_detector.predict(X)
        score_ae = float(scores_ae[0])
        score_ae = max(0.0, min(score_ae, 3.0)) / 3.0  # normalizar 0-1

        # simple decision
        is_fraud = (score_xgb >= 0.5) or (score_ae >= 0.5)

        msg = "Posible fraude" if is_fraud else "Transacción normal"

        return PredictionResponse(
            is_fraud=is_fraud,
            score_xgb=score_xgb,
            score_autoencoder=score_ae,
            message=msg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))