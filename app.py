from __future__ import annotations
import os
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

MODEL_PATH = os.getenv("MODEL_PATH", "heart_disease_pipeline.joblib")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


class HeartFeatures(BaseModel):
    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="0=female, 1=male (as encoded in dataset)")
    cp: int = Field(..., description="Chest pain type (encoded)")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: int = Field(..., description="Resting ECG results (encoded)")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina (0/1)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of peak exercise ST segment (encoded)")
    ca: float = Field(..., description="Number of major vessels (0-3)")
    thal: int = Field(..., description="Thalassemia (encoded)")


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    model_path: str


_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                f"Place heart_disease_pipeline.joblib next to app.py or set MODEL_PATH env var."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        _ = get_model()
        return {"status": "ok", "model_loaded": True, "model_path": MODEL_PATH}
    except Exception as e:
        return {"status": "degraded", "model_loaded": False, "model_path": MODEL_PATH, "error": str(e)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartFeatures):
    try:
        logger.info(f"Incoming prediction request: {payload.model_dump()}")
        model = get_model()
        row = pd.DataFrame([payload.model_dump()])

        proba = float(model.predict_proba(row)[0][1])  # probability of class "1" (disease)
        pred = int(proba >= 0.5)

        logger.info(f"Prediction: {pred}, Confidence: {proba:.4f}")

        return PredictionResponse(prediction=pred, confidence=proba, model_path=MODEL_PATH)

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Loaded model does not support predict_proba. Ensure you saved the sklearn Pipeline correctly.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
