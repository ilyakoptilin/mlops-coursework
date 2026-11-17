from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_coursework.api import app
from mlops_coursework.features import (
    ALL_FEATURE_COLUMNS,
    build_features,
    load_config,
    prepare_raw_dataframe,
)

client = TestClient(app)


def test_feature_engineering_unit() -> None:
    sample = pd.DataFrame(
        [
            {
                "machine_type": "M",
                "air_temp_k": 300.0,
                "process_temp_k": 310.0,
                "rotational_speed_rpm": 1500,
                "torque_nm": 40.0,
                "tool_wear_min": 50,
            }
        ]
    )
    features = build_features(sample)
    assert list(features.columns) == ALL_FEATURE_COLUMNS
    assert features["temp_diff_k"].iloc[0] == pytest.approx(10.0)
    assert features["overstrain_index"].iloc[0] == pytest.approx(2000.0)
    assert features["power_w"].iloc[0] == pytest.approx(6283.1853, rel=1e-4)


def test_prepare_raw_dataframe_unit() -> None:
    raw = pd.DataFrame(
        [
            {
                "UDI": 1,
                "Product ID": "M14860",
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0,
                "Machine failure": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        ]
    )
    processed = prepare_raw_dataframe(raw)
    assert "target" in processed.columns
    assert "TWF" not in processed.columns
    assert processed["target"].iloc[0] == 0


def test_config_validation_unit(tmp_path: Path) -> None:
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("train:\n  test_size: 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inference"):
        load_config(bad_cfg)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_predict_normal_operation() -> None:
    payload = {
        "machine_type": "M",
        "air_temp_k": 298.1,
        "process_temp_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 10,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["failure_prediction"] == 0
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["risk_level"] == "low"
    assert data["model_version"] == "v1.0.0"


def test_predict_high_risk_failure() -> None:
    payload = {
        "machine_type": "L",
        "air_temp_k": 302.5,
        "process_temp_k": 310.1,
        "rotational_speed_rpm": 1320,
        "torque_nm": 68.5,
        "tool_wear_min": 225,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["failure_prediction"] == 1
    assert data["failure_probability"] > 0.5
    assert data["risk_level"] in {"medium", "high"}


def test_predict_validation_error() -> None:
    bad_payload = {
        "machine_type": "UNKNOWN",
        "air_temp_k": -10.0,
        "process_temp_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 10,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
