import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_coursework.features import build_features


@pytest.fixture(scope="session", autouse=True)
def ensure_test_model(tmp_path_factory: pytest.TempPathFactory) -> None:
    default_model = Path("models/model.joblib")
    if default_model.exists():
        return

    tmp_dir = tmp_path_factory.mktemp("models")
    fallback_path = tmp_dir / "model.joblib"

    raw_samples = pd.DataFrame(
        [
            {
                "machine_type": "M",
                "air_temp_k": 298.1,
                "process_temp_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 10,
            },
            {
                "machine_type": "H",
                "air_temp_k": 298.5,
                "process_temp_k": 309.0,
                "rotational_speed_rpm": 1500,
                "torque_nm": 40.0,
                "tool_wear_min": 20,
            },
            {
                "machine_type": "L",
                "air_temp_k": 302.5,
                "process_temp_k": 310.1,
                "rotational_speed_rpm": 1320,
                "torque_nm": 68.5,
                "tool_wear_min": 225,
            },
            {
                "machine_type": "L",
                "air_temp_k": 303.0,
                "process_temp_k": 311.0,
                "rotational_speed_rpm": 1300,
                "torque_nm": 70.0,
                "tool_wear_min": 230,
            },
        ]
    )
    x_train = build_features(raw_samples)
    y_train = [0, 0, 1, 1]

    cat_cols = ["machine_type"]
    num_cols = [c for c in x_train.columns if c != "machine_type"]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingClassifier(min_samples_leaf=1)),
        ]
    )
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, fallback_path)

    os.environ["MODEL_PATH"] = str(fallback_path)
    from mlops_coursework import api

    api.MODEL_PATH = fallback_path
