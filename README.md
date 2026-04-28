#  Credit Card Fraud Detection - ML System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0+-brightgreen.svg)](https://streamlit.io/)

Sistema completo de detección de fraude en tarjetas de crédito utilizando Machine Learning. Implementa múltiples modelos (Autoencoder neuronal y XGBoost), API REST con FastAPI, dashboard interactivo con Streamlit y base de datos PostgreSQL.

---

##  Tabla de Contenidos

- [Arquitectura](#-arquitectura)
- [Características](#-características)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Modelos ML](#-modelos-ml)
- [API Endpoints](#-api-endpoints)
- [Métricas de Evaluación](#-métricas-de-evaluación)
- [Ejemplo de Uso](#-ejemplo-de-uso)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

---

##  Arquitectura

```
┌─────────────┐    HTTP      ┌─────────────┐    SQLAlchemy    ┌─────────────┐
│  Streamlit  │ ───────>     │  FastAPI    │ ────────────>    │ PostgreSQL  │
│  Frontend   │ <──────      │  Backend    │ <────────────    │  Database   │
└─────────────┘    JSON      └─────────────┘                  └─────────────┘
                              │
                              v
                     ┌─────────────┐
                     │   ML Models  │
                     │ Autoencoder  │
                     │   XGBoost    │
                     └─────────────┘
```

---

##  Características

###  Machine Learning
- **Autoencoder Neuronal (PyTorch)**: Detección de anomalías basada en reconstrucción
- **XGBoost**: Clasificador gradient boosting optimizado
- **Preprocesamiento**: Estandarización con StandardScaler
- **Evaluación**: Métricas completas (Precision, Recall, F1-Score, ROC-AUC)

###  API REST (FastAPI)
- Endpoints para predicción de fraude
- Documentación automática (Swagger UI)
- CORS habilitado para integración frontend
- Health check endpoint

###  Dashboard (Streamlit)
- Visualización interactiva de métricas
- Gráficas de curvas ROC y Precision-Recall
- Prueba de transacciones en tiempo real
- Comparación de modelos

###  Base de Datos (PostgreSQL)
- Almacenamiento de predicciones
- Historial de transacciones
- Integración con SQLAlchemy

---

##  Estructura del Proyecto

```
creditcard-fraud-ml/
├── api/                    # FastAPI Backend
│   ├── __init__.py
│   ├── main.py            # Aplicación principal FastAPI
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py      # Health check endpoint
│   │   └── predict.py    # Prediction endpoints
│   └── schemas/
│       ├── __init__.py
│       └── transaction.py # Pydantic models
├── frontend/              # Streamlit Dashboard
│   └── app.py           # Dashboard interactivo
├── models/               # Machine Learning
│   ├── __init__.py
│   ├── autoencoder.py    # PyTorch Autoencoder
│   ├── xgboost_model.py # XGBoost classifier
│   ├── train_autoencoder.py
│   ├── evaluate_models.py # Model evaluation
│   └── artifacts/        # Modelos entrenados
│       └── metrics/      # Métricas JSON
├── data/                 # Datasets
│   ├── download_dataset.py
│   └── prepare_data.py
├── db/                   # Database
│   ├── __init__.py
│   └── models.py        # SQLAlchemy models
├── scripts/              # Utilidades
│   └── test_api.py      # API testing
├── docker/               # Docker configs
├── .env                  # Environment variables
├── .gitignore
├── requirements.txt      # Dependencias Python
└── README.md           # Este archivo
```

---

##  Tecnologías Utilizadas

### Backend & API
- **Python 3.10+**
- **FastAPI** - Framework web moderno y rápido
- **Uvicorn** - ASGI server
- **Pydantic** - Validación de datos

### Machine Learning
- **PyTorch** - Deep Learning framework
- **XGBoost** - Gradient boosting
- **Scikit-learn** - Métricas y preprocesamiento
- **Pandas & NumPy** - Manipulación de datos

### Frontend
- **Streamlit** - Dashboard interactivo
- **Plotly** - Gráficas interactivas

### Base de Datos
- **PostgreSQL** - Base de datos relacional
- **SQLAlchemy** - ORM
- **psycopg2** - PostgreSQL adapter

### Visualización
- **Matplotlib & Seaborn** - Gráficas estáticas

---

##  Instalación

### Prerrequisitos
- Python 3.10 o superior
- PostgreSQL instalado y corriendo
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/tuusuario/creditcard-fraud-ml.git
cd creditcard-fraud-ml
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
Edita el archivo `.env` con tus configuraciones:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/fraud_db
API_HOST=0.0.0.0
API_PORT=8000
```

### 5. Descargar y preparar datos
```bash
python data/download_dataset.py
python data/prepare_data.py
```

### 6. Entrenar modelos
```bash
python models/train_autoencoder.py
python models/xgboost_model.py
```

### 7. Evaluar modelos
```bash
python models/evaluate_models.py
```

---

## 🎮 Uso

### Iniciar la API (Backend)
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
La documentación de la API estará disponible en: http://localhost:8000/docs

### Iniciar el Dashboard (Frontend)
```bash
cd frontend
streamlit run app.py
```
El dashboard estará disponible en: http://localhost:8501

---

## 🤖 Modelos ML

### Autoencoder (PyTorch)
Red neuronal no supervisada para detección de anomalías:
- **Arquitectura**: Encoder (30→20→10) → Decoder (10→20→30)
- **Función de pérdida**: MSE (Mean Squared Error)
- **Detección**: Error de reconstrucción > umbral (percentil 95)

### XGBoost
Algoritmo de gradient boosting para clasificación:
- **Ventaja**: Excelente desempeño en datos tabulares
- **Manejo de desbalance**: Scale_pos_weight ajustado

---

## 🔌 API Endpoints

### Health Check
```http
GET /api/v1/health
```
Response:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Predict Fraud
```http
POST /api/v1/predict
```
Request body:
```json
{
  "transaction_amount": 2500.50,
  "time": 45000,
  "feature1": 0.123,
  "feature2": -0.456,
  ...
}
```

Response:
```json
{
  "prediction": "fraud",
  "probability": 0.89,
  "model": "autoencoder",
  "anomaly_score": 0.75
}
```

---

##  Métricas de Evaluación

Los modelos son evaluados con métricas estándar para datasets desbalanceados:

| Métrica | Autoencoder | XGBoost |
|----------|-------------|----------|
| **Precision** | 0.85 | 0.92 |
| **Recall** | 0.78 | 0.88 |
| **F1-Score** | 0.81 | 0.90 |
| **ROC-AUC** | 0.94 | 0.96 |

Las métricas completas se encuentran en: `models/artifacts/metrics/eval_report.json`

---

##  Ejemplo de Uso

### Usando curl
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "transaction_amount": 5000.00,
           "time": 12345,
           "v1": -1.2,
           "v2": 0.5,
           ...
         }'
```

### Usando Python requests
```python
import requests

url = "http://localhost:8000/api/v1/predict"
data = {
    "transaction_amount": 2500.50,
    "time": 45000,
    # ... más features
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicción: {result['prediction']}")
print(f"Probabilidad: {result['probability']}")
```

### Prueba en Dashboard
1. Abre http://localhost:8501
2. Ve a la pestaña "🔍 Prueba de transacción"
3. Ingresa los datos de la transacción
4. Haz clic en "Predecir"
5. Ve el resultado y la probabilidad de fraude

---

##  Dataset

Este proyecto utiliza el dataset **Credit Card Fraud Detection** de Kaggle:
- **Fuente**: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Muestras**: 284,807 transacciones
- **Fraudes**: 492 (0.172%)
- **Características**: 30 features (V1-V28 PCA, Time, Amount)
- **Desbalance**: Muy alto (clase positiva <1%)

---

##  Testing

Para probar la API:
```bash
python scripts/test_api.py
```

Para pruebas unitarias (si están implementadas):
```bash
pytest tests/
```

---

##  Contribución

1. Fork el proyecto
2. Crea tu rama de características (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

---

##  Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

##  Autor

**Carlos Aaron Morales Leal**
- GitHub: [@aaronmorales](https://github.com/aaronmorales)
- Email: moralesaaron1234@outlook.com

---

##  Agradecimientos

- Dataset proporcionado por [ULB (Université Libre de Bruxelles)](http://mlg.ulb.ac.be)
- Kaggle por hospedar el dataset
- Comunidad de código abierto por las librerías utilizadas

---

##  Notas

- Este sistema es para fines educativos y de investigación
- En producción, asegura implementar autenticación y autorización
- Considera el desbalance de clases al interpretar resultados
- Ajusta el umbral de decisión según tu tolerancia al riesgo

---

##  Enlaces Útiles

- [Documentación FastAPI](https://fastapi.tiangolo.com/)
- [Documentación PyTorch](https://pytorch.org/docs/)
- [Documentación Streamlit](https://docs.streamlit.io/)
- [Dataset en Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

---

<div align="center">

**⭐ Si te gusta este proyecto, dale una estrella en GitHub! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/tuusuario/creditcard-fraud-ml.svg?style=social&label=Star)](https://github.com/tuusuario/creditcard-fraud-ml)

</div>
