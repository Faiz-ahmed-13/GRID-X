"""
GRID-X Prediction API
Uses the integrated predictor int_en_pred_2.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import io
import numpy as np
import pandas as pd

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
from torchvision import transforms
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent.parent  # GRID-X folder
sys.path.append(str(project_root))

# Import your integrated predictor and strategy optimiser
from scripts.models.int_en_pred_2 import GridXIntegratedPredictor
from scripts.models.strategy_optimiser import StrategyOptimiser

# ---------- CNN model imports (dynamic, to avoid package issues) ----------
cnn_model_path = project_root / 'scripts' / 'models' / 'cnn' / 'model.py'
if cnn_model_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cnn_model", cnn_model_path)
    cnn_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn_model)
    get_model = cnn_model.get_model
else:
    get_model = None
    print("⚠️ CNN model module not found – /analyze-circuit disabled.")

# ========== Pydantic Models ==========
class Conditions(BaseModel):
    air_temp: float = Field(..., description="Air temperature in °C")
    track_temp: float = Field(..., description="Track temperature in °C")
    humidity: float = Field(..., description="Humidity in %")
    rainfall: float = Field(..., description="Rainfall intensity (0 = dry, 0.5 = light, 1 = heavy)")

class RaceInput(BaseModel):
    circuit: str
    year: int
    qualifying_results: Dict[str, int] = Field(..., description="Driver code -> grid position")
    conditions: Conditions
    tyre_compounds: Dict[str, str] = Field(..., description="Driver code -> compound (SOFT, MEDIUM, HARD, etc.)")
    round: Optional[int] = 1

class StintRequest(BaseModel):
    driver_code: str
    circuit: str
    compound: str
    weather: Conditions
    n_laps: int = Field(20, ge=1, le=50, description="Number of laps to simulate (default 20)")

class StrategyRequest(BaseModel):
    driver: str
    circuit: str
    weather: Conditions
    total_laps: int = Field(..., ge=10, le=100, description="Total race laps")
    start_compound: str = Field("SOFT", description="Starting tyre compound")

class LapFeatures(BaseModel):
    DriverNumber: str
    LapNumber: int
    Stint: int
    Compound: str
    Team: str
    event_name: str
    circuit: str
    year: int
    round: int
    AirTemp: float
    Humidity: float
    Pressure: float = 1013.0
    Rainfall: float
    TrackTemp: float
    WindSpeed: float = 5.0
    WindDirection: float = 180.0
    stint_lap_number: int
    tyre_age_laps: int
    session_progress: float
    Position: int
    position_change: int = 0
    AggressionScore: float
    ConsistencyScore: float
    BrakingIntensity: float
    TyrePreservation: float
    OvertakingAbility: float

    @validator('session_progress')
    def session_progress_range(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('session_progress must be between 0 and 1')
        return v

class NextLapRequest(BaseModel):
    laps: List[LapFeatures] = Field(..., min_items=10, max_items=10, description="Exactly 10 laps of history")

# NEW: Pydantic model for explainability endpoint
class LapExplanationRequest(BaseModel):
    driver: str
    circuit: str
    weather: Conditions
    compound: str = Field(..., description="Tyre compound (SOFT, MEDIUM, HARD)")
    lap_number: int = Field(1, ge=1, description="Lap number")
    stint: int = Field(1, ge=1, description="Stint number")
    tyre_age: int = Field(1, ge=1, description="Laps on current tyres")
    session_progress: float = Field(0.05, ge=0, le=1, description="Race progress (0-1)")
    total_laps: int = Field(20, ge=1, description="Total race laps")

# ========== FastAPI App ==========
app = FastAPI(
    title="GRID-X Prediction API",
    description="Formula 1 race prediction using trained ML models",
    version="1.4.0"  # bumped version for explainability
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor = None
circuit_model = None
class_names = []
circuit_metadata = None
circuit_embeddings = None
circuit_labels = None
similarity_transform = None

@app.on_event("startup")
async def load_predictor():
    global predictor, circuit_model, class_names, circuit_metadata
    global circuit_embeddings, circuit_labels, similarity_transform

    print("🔄 Loading GRID-X models...")
    predictor = GridXIntegratedPredictor()
    predictor.load_or_train_models()
    print("✅ Main models loaded.")

    # ---------- Load CNN circuit classifier ----------
    model_path = project_root / 'models' / 'circuit_classifier.pth'
    class_names_path = project_root / 'models' / 'class_names.txt'
    if get_model is not None and model_path.exists() and class_names_path.exists():
        with open(class_names_path) as f:
            class_names = [line.strip() for line in f]
        circuit_model = get_model(num_classes=len(class_names), pretrained=False)
        circuit_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        circuit_model.eval()
        print(f"✅ CNN circuit classifier loaded ({len(class_names)} classes).")

        # Load pre‑computed embeddings for similarity
        emb_path = project_root / 'models' / 'circuit_embeddings.npy'
        lbl_path = project_root / 'models' / 'circuit_labels.npy'
        if emb_path.exists() and lbl_path.exists():
            circuit_embeddings = np.load(emb_path)
            circuit_labels = np.load(lbl_path)
            print(f"✅ Circuit embeddings loaded: {circuit_embeddings.shape}")
        else:
            print("⚠️ Embeddings not found – similarity search disabled.")
    else:
        circuit_model = None
        class_names = []
        print("⚠️ CNN circuit classifier not found – /analyze-circuit disabled.")

    # ---------- Load circuit metadata ----------
    csv_path = project_root / 'data' / 'cnn' / 'circuit_metadata.csv'
    if csv_path.exists():
        # Custom reading to handle mixed delimiter (space-separated header, comma-separated data)
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            col_names = first_line.split()  # split on whitespace
            # Read the rest of the file with pandas, treating as comma-separated
            circuit_metadata = pd.read_csv(csv_path, skiprows=1, header=None, names=col_names, encoding='utf-8')
        circuit_metadata.columns = circuit_metadata.columns.str.strip()
        # Ensure 'circuit_name' column exists
        if 'circuit_name' not in circuit_metadata.columns:
            # try to find an alternative
            possible = ['circuit name', 'CircuitName', 'circuit']
            for p in possible:
                if p in circuit_metadata.columns:
                    circuit_metadata.rename(columns={p: 'circuit_name'}, inplace=True)
                    break
        circuit_metadata = circuit_metadata.set_index('circuit_name')
        print("✅ Circuit metadata loaded.")
    else:
        circuit_metadata = None
        print("⚠️ Circuit metadata not found.")

    # ---------- Image transform for CNN ----------
    similarity_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@app.get("/")
async def root():
    return {
        "message": "GRID-X Prediction API",
        "status": "running",
        "docs": "/docs"
    }

@app.post("/predict")
async def predict_race(race: RaceInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        race_dict = race.dict()
        result = predictor.integrated_race_prediction(race_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stint-simulate")
async def stint_simulate(request: StintRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        laps = predictor.simulate_stint(
            driver_code=request.driver_code,
            circuit=request.circuit,
            compound=request.compound,
            weather=request.weather.dict(),
            n_laps=request.n_laps
        )
        return {
            "driver": request.driver_code,
            "circuit": request.circuit,
            "compound": request.compound,
            "weather": request.weather.dict(),
            "n_laps": request.n_laps,
            "laps": laps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/next-lap")
async def predict_next_lap(request: NextLapRequest):
    """
    Predict the next lap time given the last 10 laps of data.
    Each lap must include all features required by the LSTM model,
    including driver style scores (AggressionScore, ConsistencyScore, etc.).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        history_df = pd.DataFrame([lap.dict() for lap in request.laps])
        next_lap = predictor.predict_next_lap(history_df)
        return {"predicted_next_lap": round(next_lap, 3)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategy-optimize")
async def strategy_optimize(request: StrategyRequest):
    """
    Use the trained DQN agent to recommend an optimal pit strategy for a given race.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        optimiser = StrategyOptimiser()
        optimiser.load()
        result = optimiser.optimize(
            driver=request.driver,
            circuit=request.circuit,
            weather=request.weather.dict(),
            total_laps=request.total_laps,
            start_compound=request.start_compound
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Circuit Analysis Endpoint ==========
@app.post("/analyze-circuit")
async def analyze_circuit(file: UploadFile = File(...)):
    """
    Upload a circuit image (PNG, JPG) and receive:
    - Top‑3 circuit predictions with confidence scores.
    - Metadata for the top prediction.
    - Up to 5 visually similar circuits.
    """
    if circuit_model is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded")

    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = similarity_transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = circuit_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, 3)

        # Build predictions with metadata
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            circuit_name = class_names[idx]
            meta = {}
            if circuit_metadata is not None and circuit_name in circuit_metadata.index:
                meta = circuit_metadata.loc[circuit_name].to_dict()
            predictions.append({
                "circuit": circuit_name,
                "confidence": round(prob.item(), 3),
                "metadata": meta
            })

        # ---------- Similar circuits (optional) ----------
        similar = []
        if circuit_embeddings is not None and circuit_labels is not None:
            # Extract embedding of the query image
            with torch.no_grad():
                x = circuit_model.conv1(image_tensor)
                x = circuit_model.bn1(x)
                x = circuit_model.relu(x)
                x = circuit_model.maxpool(x)
                x = circuit_model.layer1(x)
                x = circuit_model.layer2(x)
                x = circuit_model.layer3(x)
                x = circuit_model.layer4(x)
                x = circuit_model.avgpool(x)
                query_emb = torch.flatten(x, 1).cpu().numpy()

            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_emb, circuit_embeddings)[0]
            top_sim_idx = np.argsort(similarities)[-5:][::-1]
            for idx in top_sim_idx:
                sim_circuit = class_names[circuit_labels[idx]]
                if sim_circuit == predictions[0]["circuit"]:
                    continue
                similar.append({
                    "circuit": sim_circuit,
                    "similarity_score": round(float(similarities[idx]), 3)
                })

        return {
            "success": True,
            "predictions": predictions,
            "similar_circuits": similar[:5]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== NEW: Explainability Endpoint ==========
@app.post("/explain-lap")
async def explain_lap(request: LapExplanationRequest):
    """
    Explain a lap time prediction using SHAP.
    Returns the top 10 features with their SHAP values.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Build the feature dictionary expected by the lap time model
        driver_num = predictor.driver_number_map.get(request.driver, 0)
        feat = {
            'DriverNumber': str(driver_num),
            'LapNumber': request.lap_number,
            'Stint': request.stint,
            'Compound': request.compound,
            'Team': 'UNKNOWN',
            'event_name': f"{request.circuit} Grand Prix",
            'circuit': request.circuit,
            'year': 2024,
            'round': 1,
            'AirTemp': request.weather.air_temp,
            'Humidity': request.weather.humidity,
            'Pressure': 1013.0,
            'Rainfall': request.weather.rainfall,
            'TrackTemp': request.weather.track_temp,
            'WindSpeed': 5.0,
            'WindDirection': 180,
            'stint_lap_number': request.tyre_age,
            'tyre_age_laps': request.tyre_age,
            'session_progress': request.lap_number / request.total_laps,
            'Position': 1,
            'position_change': 0
        }

        # Get SHAP explanation
        explanation = predictor.explain_lap_time(feat)

        return {"success": True, "explanation": explanation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)