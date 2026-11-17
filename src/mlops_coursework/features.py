import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RAW_COLUMN_MAPPING = {
    "Type": "machine_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "target",
}

BASE_FEATURE_COLUMNS = [
    "machine_type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

ENGINEERED_FEATURE_COLUMNS = [
    "temp_diff_k",
    "power_w",
    "overstrain_index",
]

ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS

REQUIRED_TRAIN_KEYS = {
    "raw_data_path",
    "processed_data_path",
    "model_output_path",
    "metrics_output_path",
    "test_size",
    "random_state",
}

REQUIRED_INFERENCE_KEYS = {
    "model_path",
    "model_version",
    "high_risk_threshold",
    "medium_risk_threshold",
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path or os.getenv("APP_CONFIG_PATH", "configs/config.yaml"))
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a YAML mapping.")

    for section in ("train", "inference"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing required configuration section: '{section}'")

    missing_train = REQUIRED_TRAIN_KEYS - set(config["train"].keys())
    if missing_train:
        raise ValueError(f"Missing required 'train' keys: {sorted(missing_train)}")

    missing_inf = REQUIRED_INFERENCE_KEYS - set(config["inference"].keys())
    if missing_inf:
        raise ValueError(f"Missing required 'inference' keys: {sorted(missing_inf)}")

    return config


def prepare_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(RAW_COLUMN_MAPPING.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected raw columns: {sorted(missing_cols)}")

    renamed = df[list(RAW_COLUMN_MAPPING.keys())].rename(columns=RAW_COLUMN_MAPPING).copy()
    features = build_features(renamed[BASE_FEATURE_COLUMNS])
    features["target"] = renamed["target"].astype(int)
    return features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = set(BASE_FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required input feature columns: {sorted(missing_cols)}")

    result = df[BASE_FEATURE_COLUMNS].copy()
    result["temp_diff_k"] = result["process_temp_k"] - result["air_temp_k"]
    omega_rad_s = result["rotational_speed_rpm"] * 2.0 * math.pi / 60.0
    result["power_w"] = result["torque_nm"] * omega_rad_s
    result["overstrain_index"] = result["torque_nm"] * result["tool_wear_min"]
    return result[ALL_FEATURE_COLUMNS]
