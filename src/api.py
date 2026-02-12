"""
api.py - Smart Ambulance Risk Scoring API
Run with: python api.py
"""

import sys
import os

# Add src folder to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import numpy as np
from datetime import datetime
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from preprocessing import VitalsPreprocessor, extract_features_from_vitals
from inference import RiskScorer


# ========================================
# MODEL PATHS
# ========================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ISO_FOREST_PATH = os.path.join(MODELS_DIR, "iso_forest.pkl")
OCSVM_PATH      = os.path.join(MODELS_DIR, "ocsvm.pkl")
LSTM_PATH       = os.path.join(MODELS_DIR, "lstm_autoencoder.keras")

# Global objects
preprocessor = None
risk_scorer  = None


# ========================================
# LIFESPAN (replaces deprecated on_event)
# ========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models when API starts."""
    global preprocessor, risk_scorer

    print("\n" + "="*60)
    print("  SMART AMBULANCE API - STARTING UP")
    print("="*60)

    # Check model files exist
    missing = []
    for path, name in [
        (ISO_FOREST_PATH, "iso_forest.pkl"),
        (OCSVM_PATH,      "ocsvm.pkl"),
        (LSTM_PATH,       "lstm_autoencoder.keras")
    ]:
        if not os.path.exists(path):
            missing.append(name)
            print(f"  NOT FOUND: {path}")
        else:
            print(f"  Found: {name}")

    if missing:
        print(f"\n  ERROR: Missing model files: {missing}")
        print(f"  Expected in: {os.path.abspath(MODELS_DIR)}")
    else:
        try:
            preprocessor = VitalsPreprocessor()
            print("  Preprocessor: loaded")

            risk_scorer = RiskScorer(
                iso_forest_path=ISO_FOREST_PATH,
                ocsvm_path=OCSVM_PATH,
                lstm_path=LSTM_PATH
            )
            print("  Risk Scorer:  loaded")
            print("\n  API is READY!")

        except Exception as e:
            print(f"\n  ERROR loading models: {e}")
            import traceback
            traceback.print_exc()

    print("="*60)
    print("  Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    yield  # API runs here

    print("\nShutting down API...")


# ========================================
# FASTAPI APP
# ========================================
app = FastAPI(
    title="Smart Ambulance Risk Scoring API",
    description="Real-time anomaly detection for ambulance patient vitals",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# PYDANTIC MODELS
# ========================================
class VitalsWindow(BaseModel):
    """One window of vital sign readings (typically 30 values = 30 seconds)."""
    hr:     List[float] = Field(..., description="Heart rate (bpm)")
    spo2:   List[float] = Field(..., description="SpO2 (%)")
    sbp:    List[float] = Field(..., description="Systolic BP (mmHg)")
    dbp:    List[float] = Field(..., description="Diastolic BP (mmHg)")
    motion: List[float] = Field(..., description="Motion signal")


class PredictionRequest(BaseModel):
    vitals:     VitalsWindow
    patient_id: Optional[str] = None


class BatchRequest(BaseModel):
    vitals_list: List[VitalsWindow]
    patient_id:  Optional[str] = None


class RiskPrediction(BaseModel):
    risk_score:    float
    risk_level:    str
    confidence:    float
    reasoning:     List[str]
    model_scores:  Dict[str, float]
    timestamp:     str


# ========================================
# ENDPOINTS
# ========================================

@app.get("/")
async def root():
    return {
        "name": "Smart Ambulance Risk Scoring API",
        "version": "1.0.0",
        "status": "running",
        "docs": "http://localhost:8000/docs"
    }


@app.get("/health")
async def health():
    models_loaded = preprocessor is not None and risk_scorer is not None
    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=RiskPrediction)
async def predict(request: PredictionRequest):
    if not risk_scorer:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        features = extract_features_from_vitals(
            hr=request.vitals.hr,
            spo2=request.vitals.spo2,
            sbp=request.vitals.sbp,
            dbp=request.vitals.dbp,
            motion=request.vitals.motion
        )

        result = risk_scorer.calculate_risk_score(features)

        return RiskPrediction(
            risk_score=result['risk_score'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            model_scores=result['model_scores'],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(request: BatchRequest):
    if not risk_scorer:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        results = []
        for vitals in request.vitals_list:
            features = extract_features_from_vitals(
                hr=vitals.hr,
                spo2=vitals.spo2,
                sbp=vitals.sbp,
                dbp=vitals.dbp,
                motion=vitals.motion
            )
            result = risk_scorer.calculate_risk_score(features)
            results.append({
                "risk_score":   result['risk_score'],
                "risk_level":   result['risk_level'],
                "confidence":   result['confidence'],
                "reasoning":    result['reasoning'],
                "model_scores": result['model_scores'],
                "timestamp":    datetime.now().isoformat()
            })

        return {
            "predictions":    results,
            "total_windows":  len(results),
            "patient_id":     request.patient_id
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset():
    if not risk_scorer:
        raise HTTPException(status_code=503, detail="Models not loaded")
    risk_scorer.reset_history()
    return {
        "status": "success",
        "message": "Patient session reset",
        "timestamp": datetime.now().isoformat()
    }


# ========================================
# RUN
# ========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")