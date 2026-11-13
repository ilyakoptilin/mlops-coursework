import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@pytest.fixture(autouse=True, scope="session")
def ensure_model_exists(tmp_path_factory):
    model_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    if model_path.exists():
        return

    tmp_dir = tmp_path_factory.mktemp("models")
    fallback_path = tmp_dir / "model.joblib"

    df = pd.DataFrame(
        [
            {
                "machine_type": "M",
                "air_temp_k": 298.1,
                "process_temp_k": 308.6,
                "rotational_speed_rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 10,
                "temp_diff_k": 10.5,
                "power_w": 6951.59,
                "overstrain_index": 428.0,
                "target": 0,
            },
            {
                "machine_type": "L",
                "air_temp_k": 302.5,
                "process_temp_k": 310.1,
                "rotational_speed_rpm": 1320,
                "torque_nm": 68.5,
                "tool_wear_min": 225,
                "temp_diff_k": 7.6,
                "power_w": 9468.76,
                "overstrain_index": 15412.5,
                "target": 1,
            },
        ]
        * 10
    )

    cat_cols = ["machine_type"]
    num_cols = [
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temp_diff_k",
        "power_w",
        "overstrain_index",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    min_samples_leaf=1, max_iter=50, random_state=42
                ),
            ),
        ]
    )
    pipe.fit(df[cat_cols + num_cols], df["target"])
    joblib.dump(pipe, fallback_path)
    os.environ["MODEL_PATH"] = str(fallback_path)
