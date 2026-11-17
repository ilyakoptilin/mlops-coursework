import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_coursework.features import build_features, load_config

CONFIG = load_config()
INFERENCE_CFG = CONFIG["inference"]
MODEL_PATH = Path(os.getenv("MODEL_PATH", INFERENCE_CFG["model_path"]))
MODEL_VERSION = os.getenv("MODEL_VERSION", INFERENCE_CFG["model_version"])

app = FastAPI(
    title="AI4I Predictive Maintenance API",
    description="Inference service for industrial machinery failure prediction",
    version=MODEL_VERSION,
)


class SensorInput(BaseModel):
    machine_type: Literal["L", "M", "H"] = Field(..., examples=["M"])
    air_temp_k: float = Field(..., gt=0, examples=[298.1])
    process_temp_k: float = Field(..., gt=0, examples=[308.6])
    rotational_speed_rpm: int = Field(..., gt=0, examples=[1551])
    torque_nm: float = Field(..., ge=0, examples=[42.8])
    tool_wear_min: int = Field(..., ge=0, examples=[108])


class PredictionResponse(BaseModel):
    failure_prediction: int
    failure_probability: float
    risk_level: Literal["low", "medium", "high"]
    model_version: str


def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model artifact not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SensorInput) -> PredictionResponse:
    model = get_model()
    raw_df = pd.DataFrame([payload.model_dump()])
    features_df = build_features(raw_df)

    prob = float(model.predict_proba(features_df)[0, 1])
    pred = int(model.predict(features_df)[0])

    if prob >= INFERENCE_CFG["high_risk_threshold"]:
        risk = "high"
    elif prob >= INFERENCE_CFG["medium_risk_threshold"]:
        risk = "medium"
    else:
        risk = "low"

    return PredictionResponse(
        failure_prediction=pred,
        failure_probability=round(prob, 4),
        risk_level=risk,
        model_version=MODEL_VERSION,
    )
