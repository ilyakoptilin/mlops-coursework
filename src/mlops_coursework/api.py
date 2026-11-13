import math
import os
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="AI4I Predictive Maintenance API",
    description="Сервис прогнозирования отказов промышленных станков по телеметрии датчиков",
    version=os.getenv("MODEL_VERSION", "v1.0.0"),
)

_model_cache: dict[str, object] = {}


class SensorInput(BaseModel):
    machine_type: Literal["L", "M", "H"] = Field(..., examples=["M"])
    air_temp_k: float = Field(..., gt=0, examples=[298.1])
    process_temp_k: float = Field(..., gt=0, examples=[308.6])
    rotational_speed_rpm: int = Field(..., gt=0, examples=[1551])
    torque_nm: float = Field(..., ge=0, examples=[42.8])
    tool_wear_min: int = Field(..., ge=0, examples=[108])


class PredictionOutput(BaseModel):
    failure_prediction: int
    failure_probability: float
    risk_level: str
    model_version: str


def get_model():
    model_path = os.getenv("MODEL_PATH", "models/model.joblib")
    if model_path not in _model_cache:
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Файл модели не найден по пути: {model_path}",
            )
        _model_cache[model_path] = joblib.load(model_path)
    return _model_cache[model_path]


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    model_path = os.getenv("MODEL_PATH", "models/model.joblib")
    return {
        "status": "ok",
        "model_loaded": Path(model_path).exists(),
        "model_version": os.getenv("MODEL_VERSION", "v1.0.0"),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: SensorInput) -> PredictionOutput:
    model = get_model()

    temp_diff_k = round(payload.process_temp_k - payload.air_temp_k, 2)
    power_w = round(
        payload.torque_nm * (payload.rotational_speed_rpm * 2 * math.pi / 60), 2
    )
    overstrain_index = round(payload.torque_nm * payload.tool_wear_min, 2)

    features = pd.DataFrame(
        [
            {
                "machine_type": payload.machine_type,
                "air_temp_k": payload.air_temp_k,
                "process_temp_k": payload.process_temp_k,
                "rotational_speed_rpm": payload.rotational_speed_rpm,
                "torque_nm": payload.torque_nm,
                "tool_wear_min": payload.tool_wear_min,
                "temp_diff_k": temp_diff_k,
                "power_w": power_w,
                "overstrain_index": overstrain_index,
            }
        ]
    )

    pred = int(model.predict(features)[0])
    proba = round(float(model.predict_proba(features)[0][1]), 4)

    if proba >= 0.7:
        risk_level = "high"
    elif proba >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    return PredictionOutput(
        failure_prediction=pred,
        failure_probability=proba,
        risk_level=risk_level,
        model_version=os.getenv("MODEL_VERSION", "v1.0.0"),
    )
