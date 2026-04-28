#!/bin/bash
set -e

echo "1) Preparando datos..."
python data/prepare_data.py

echo "2) Entrenando autoencoder..."
python models/train_autoencoder.py

echo "3) Entrenando XGBoost..."
python models/xgboost_model.py

echo "4) Evaluando modelos..."
python models/evaluate_models.py

echo "✅ Pipeline completado."