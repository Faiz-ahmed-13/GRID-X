# 🏎️ GRID‑X:Global Race Intelligence and Data eXchange is an AI‑Powered Formula 1 Prediction & Strategy Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-green)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Assets-yellow)](https://huggingface.co/datasets/faizprofesh/GRID-X_assets)
[![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)](https://opensource.org/licenses/MIT)

**GRID‑X** is an end‑to‑end machine learning ecosystem designed to simulate and predict the complexities of Formula 1 racing. It integrates **seven distinct ML paradigms** into a unified FastAPI backend, providing race predictions, strategy optimization, driver analysis, and safety assessments.

---

## ✨ Key Features

- **Lap‑Time Prediction** – Random Forest regressor with **97% accuracy** (RMSE 2.787s, R² 0.993) using 21 telemetry and environmental features.
- **Race Outcome Classification** – Dual‑era XGBoost ensemble achieving **96% win accuracy** and 93% podium accuracy.
- **Driver Style Analysis** – Unsupervised KMeans clustering categorizing drivers into **Aggressive, Smooth, Opportunistic, or Balanced** archetypes.
- **Pace Forecaster (LSTM)** – Time‑series prediction of upcoming lap times based on a 10‑lap sliding window (error ~0.12s).
- **Strategy Optimizer (RL)** – Deep Q‑Network (DQN) agent trained in a custom gymnasium environment to optimize pit‑stop windows and tyre compounds.
- **Circuit Recognition (CNN)** – ResNet50 transfer learning model that identifies F1 tracks from schematic images (97.6% accuracy on test).
- **Tire Safety Risk** – Heuristic + ML model assessing tire degradation and providing safety recommendations.
- **Crash Risk Prediction** – XGBoost classifier estimating race crash probability (ROC‑AUC 0.80) and safety car likelihood.
- **Explainability (SHAP)** – Integrated SHAP values to provide feature‑level transparency for every prediction.

---

## 🏗️ Scalable Architecture

To maintain a lightweight and professional repository, this project utilizes a dual‑storage strategy:
- **GitHub:** Hosts the core logic, API endpoints, and preprocessing pipelines.
- **Hugging Face Hub:** Serves as the artifact store for the **~9GB** telemetry dataset and **1.3GB** of serialized `.joblib` and `.pth` models.

All large assets (data and models) are automatically downloaded via the Hugging Face CLI – no manual copying needed.

---

## 📁 Project Structure

```
GRID-X/
├── api/
│   └── main.py                 # FastAPI backend server with all endpoints
├── data/                       # [STORED ON HUGGING FACE]
│   ├── processed/              # ~8GB cleaned telemetry (2021‑2024)
│   └── cnn/                    # 24 circuit schematic images + metadata
├── models/                      # [STORED ON HUGGING FACE]
│   ├── lap_time_predictor.joblib (650MB)
│   ├── crash_risk_classifier.pkl
│   ├── tire_safety_model.joblib
│   ├── circuit_classifier.pth   # CNN weights
│   └── ...                     # All other pre‑trained models & scalers
├── scripts/
│   └── models/                 # Core ML logic (XGBoost, RL, CNN, LSTM, safety)
│       ├── int_en_pred_2.py     # Integrated predictor
│       ├── dqn_agent.py
│       ├── race_env.py          # RL environment
│       ├── tire_safety.py
│       └── crash_predictor/     # Crash risk module
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & Environment

```bash
git clone https://github.com/Faiz-ahmed-13/GRID-X.git
cd GRID-X
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Synchronize Assets (Data & Models)

Since the assets exceed GitHub's size limits, use the Hugging Face CLI to download the complete dataset and pre‑trained models directly into your local project:

```bash
# Install the sync tool (if not already)
pip install huggingface_hub

# Download the latest assets directly into the project root
huggingface-cli download faizprofesh/GRID-X_assets --local-dir . --repo-type dataset
```

This will create the `data/` and `models/` folders with all necessary files.

### 3. Start the API

```bash
python api/main.py
```

Access the interactive documentation at **`http://localhost:8000/docs`**.

---

## 📡 API Endpoints

| Endpoint | Method | Description | Example Visualisation |
|----------|--------|-------------|------------------------|
| `/predict` | POST | Full race prediction (lap times, driver styles, win/podium/points probabilities) | Driver cards, radar charts, probability bars |
| `/stint-simulate` | POST | Linear stint simulation & tyre degradation | Lap‑time line chart |
| `/next-lap` | POST | LSTM‑based pace forecasting (needs 10‑lap history) | Single numeric value |
| `/strategy-optimize` | POST | RL‑driven pit stop optimization (total time, pit laps, compounds) | Pit‑stop timeline |
| `/analyze-circuit` | POST | CNN track recognition from images (returns top‑3 predictions + metadata) | Image preview, top‑3 list, metadata cards |
| `/explain-lap` | POST | SHAP feature importance for a specific lap | Horizontal bar chart of top 10 features |
| `/tire-safety-check` | POST | Tire risk assessment (score, category, safe laps, SHAP factors) | Risk gauge, action badge, explanation list |
| `/crash-risk-predict` | POST | Crash probability & safety car likelihood | Probability gauge, risk level, factor list |

---

## 🧠 Machine Learning Paradigms Covered

| Paradigm | Module(s) |
|----------|-----------|
| **Supervised (Regression)** | Lap‑Time Predictor |
| **Supervised (Classification)** | Race Outcome Classifier, Crash Risk Predictor |
| **Unsupervised** | Driver Style Analyzer (KMeans) |
| **Reinforcement Learning** | Strategy Optimizer (DQN) |
| **Ensemble Learning** | Race Outcome Classifier (dual‑era XGBoost + logistic) |
| **Computer Vision** | Circuit Analyzer (ResNet50 with transfer learning) |
| **Explainability** | SHAP for lap‑time and tire safety predictions |

---

## 📊 Datasets

- **Modern Telemetry (2021–2024):** ~91,000 lap records via **FastF1**.
- **Historical Data (1950–2020):** ~27,000 entries from the **Ergast/Kaggle F1 Dataset**.
- **Vision Data:** Manual collection of 24 high‑resolution circuit schematics with metadata.

All datasets are available on Hugging Face – see [faizprofesh/GRID-X_assets](https://huggingface.co/datasets/faizprofesh/GRID-X_assets).

---

## 🤝 Contributors

- [Faiz Ahmed](https://github.com/Faiz-ahmed-13)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

*For any issues, please open a GitHub ticket or contact the contributors.*
