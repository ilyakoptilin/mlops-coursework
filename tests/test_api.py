from fastapi.testclient import TestClient
from mlops_coursework.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_normal_operation():
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


def test_predict_high_risk_failure():
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


def test_predict_validation_error():
    bad_payload = {
        "machine_type": "UNKNOWN",
        "air_temp_k": -10,
        "process_temp_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 10,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
