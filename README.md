```markdown
# 🏎️ GRID-X: AI-Powered Formula 1 Prediction & Strategy Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129.0-green)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Assets-yellow)](https://huggingface.co/datasets/faizprofesh/GRID-X_assets)
[![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)](https://opensource.org/licenses/MIT)

**GRID-X** is an end‑to‑end machine learning ecosystem designed to simulate and predict the complexities of Formula 1 racing. It leverages a **Hybrid Cloud Architecture**, separating high-performance Python logic (GitHub) from massive 9GB+ datasets and model artifacts (Hugging Face).

---

## ✨ Key Features

- **Lap‑Time Prediction** – Random Forest regressor with **97% accuracy** (RMSE 2.787s, R² 0.993) using 21 telemetry and environmental features.
- **Race Outcome Classification** – Dual‑era XGBoost ensemble achieving **96% win accuracy**.
- **Driver Style Analysis** – Unsupervised KMeans clustering categorizing drivers into Aggressive, Smooth, Opportunistic, or Balanced archetypes.
- **Pace Forecaster (LSTM)** – Time-series prediction of upcoming lap times based on a 10-lap sliding window.
- **Strategy Optimizer (RL)** – Deep Q-Network (DQN) agent trained in a custom gymnasium environment to optimize pit-stop windows and tyre compounds.
- **Circuit Recognition (CNN)** – ResNet50 transfer learning model that identifies F1 tracks from schematic images.
- **Crash Risk Prediction (XGBoost)** – Pre-race crash probability analysis using circuit characteristics, driver aggression scores, and weather factors.
- **Tire Safety Monitoring** – Real-time tire degradation assessment with risk categorization (SAFE/CAUTION/CRITICAL) and pit recommendations.
- **Explainability (SHAP)** – Integrated SHAP values to provide feature-level transparency for every prediction.

---

## 🖥️ Frontend Dashboard

**Complete responsive web interface** with 20+ interactive pages connecting directly to the FastAPI backend:

```
frontend/
├── index.html              # Main dashboard & navigation
├── pitwall.html            # Live race analytics dashboard
├── pages/
│   ├── predict_race.html      # Full race simulation
│   ├── crash_risk.html        # Pre-race crash probability
│   ├── tire_safety.html       # Real-time tire degradation
│   ├── stint_simulate.html    # Multi-lap stint prediction
│   ├── strategy_optimize.html # Optimal pit strategy
│   ├── analyze_circuit.html   # CNN circuit recognition
│   ├── explain_lap.html       # SHAP lap time explanations
│   ├── predict_next_lap.html  # Next lap time forecasting
│   └── circuit_metadata.csv   # Circuit database
├── About_us.html           # Project showcase
└── login.html              # Authentication (future)
```

**Features:** Real-time predictions, interactive charts, driver comparisons, mobile-responsive design.

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

Since the assets exceed GitHub's size limits, use the Hugging Face CLI to mirror the 9GB+ data/model directory into your local project:

```bash
# Install the sync tool
pip install huggingface_hub

# Download the latest assets directly into the root folder
huggingface-cli download faizprofesh/GRID-X_assets --local-dir . --repo-type dataset
```

### 3. Start the API Server + Frontend

```bash
# Terminal 1: Start API (loads all ML models)
python main.py
# API running at: http://localhost:8000/docs

# Terminal 2 / Browser: Open frontend
# Double-click: frontend/index.html
# OR: Open file://path/to/GRID-X/frontend/index.html
```

**Live Demo Flow:**
1. Terminal: `python main.py` → ✅ All models loaded
2. Browser: `frontend/index.html` → 🏎️ Full F1 dashboard  
3. Click any prediction → ⚡ Real-time API response

---

## 🏗️ Scalable Architecture

To maintain a lightweight and professional repository, this project utilizes a dual-storage strategy:
* **GitHub:** Hosts the core logic, API endpoints, preprocessing pipelines, **and complete frontend**.
* **Hugging Face Hub:** Serves as the artifact store for the **7.7GB** telemetry dataset and **1.4GB** of serialized `.joblib` and `.pth` models.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/predict` | POST | Full race prediction (RaceInput JSON) |
| `/stint-simulate` | POST | Linear stint simulation & tyre degradation |
| `/crash-risk-predict` | POST | Crash probability analysis |
| `/tire-safety-predict` | POST | Tire degradation risk assessment |
| `/next-lap` | POST | LSTM-based pace forecasting |
| `/strategy-optimize` | POST | RL-driven pit stop optimization |
| `/analyze-circuit` | POST | CNN track recognition from images |
| `/explain-lap` | POST | SHAP feature importance analysis |

---

## 📁 Project Structure

```bash
GRID-X/
├── API/                    # Original backend folder                 # FastAPI backend server (root level)
    ├── main.py
├── frontend/               # Complete web UI (20+ pages)
│   ├── index.html
│   ├── pitwall.html
│   └── pages/             # Feature-specific pages
├── data/                   # [STORED ON HUGGING FACE]
├── models/                 # [STORED ON HUGGING FACE]
├── scripts/
│   └── models/            # Core ML logic (XGBoost, RL, CNN, LSTM)
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

* **Modern Telemetry (2021–2024):** ~91,000 lap records via **FastF1**.
* **Historical Data (1950–2020):** ~27,000 entries from the **Ergast/Kaggle F1 Dataset**.
* **Vision Data:** Manual collection of high-resolution circuit schematics.

---

## 🤝 Contributors

* [Faiz Ahmed](https://github.com/Faiz-ahmed-13)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

**GRID-X is a complete full-stack F1 AI platform - backend + frontend + production-ready deployment!(soon)** 🏁
