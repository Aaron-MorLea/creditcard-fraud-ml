# frontend/app.py
import os
import json
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET_DIR = os.path.join(BASE_DIR, "models", "artifacts", "metrics")
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Credit Card Fraud Detection - ML System")

# Métricas
with st.sidebar:
    st.header("API")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API online")
        else:
            st.warning("API responde con error")
    except:
        st.error("No se pudo conectar a la API")

# Cargar métricas
report_path = os.path.join(MET_DIR, "eval_report.json")
if os.path.exists(report_path):
    with open(report_path) as f:
        report = json.load(f)
else:
    st.error("No se encuentra eval_report.json. Corre evaluate_models.py primero.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Métricas", "📈 Curvas", "🔍 Prueba de transacción"])

with tab1:
    st.subheader("Resumen de métricas (fraude = clase positiva)")
    rows = []
    for name, m in report.items():
        rows.append({
            "Modelo": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
        })
    df_metrics = st.dataframe(rows)

    st.markdown("**Interpretación:**")
    st.markdown("- Recall alto = pocos fraudes que se escapan (FN).")
    st.markdown("- Precision alta = pocas alertas falsas (FP).")
    st.markdown("- PR-AUC es muy importante en datos desbalanceados como fraude.")

with tab2:
    st.subheader("Curvas ROC y Precision-Recall")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ROC Curves**")
        for name in ["xgb", "autoencoder"]:
            img_path = os.path.join(MET_DIR, f"roc_{name}.png")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"ROC - {name}")
    with col2:
        st.markdown("**Precision-Recall Curves**")
        for name in ["xgb", "autoencoder"]:
            img_path = os.path.join(MET_DIR, f"pr_{name}.png")
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"PR - {name}")

    st.subheader("Matrices de confusión")
    cm_path = os.path.join(MET_DIR, "confusion_matrices.png")
    if os.path.exists(cm_path):
        st.image(Image.open(cm_path))

with tab3:
    st.subheader("Enviar transacción manual a la API")

    amount = st.number_input("Monto", min_value=0.0, value=100.0)
    time = st.number_input("Time", value=0.0)
    cols = st.columns(4)
    features = {}
    for i in range(1,29):
        with cols[(i-1) % 4]:
            features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

    if st.button("Analizar transacción"):
        payload = {
            "amount": amount,
            "time": time,
            **features
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                res = resp.json()
                if res["is_fraud"]:
                    st.error(f"⚠️ Posible fraude | XGB: {res['score_xgb']:.3f} | AE: {res['score_autoencoder']:.3f}")
                else:
                    st.success(f"✅ Transacción normal | XGB: {res['score_xgb']:.3f} | AE: {res['score_autoencoder']:.3f}")
            else:
                st.error(f"Error API: {resp.text}")
        except Exception as e:
            st.error(f"Error al llamar a la API: {e}")