import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = ["machine_type"]
NUMERIC_COLS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_w",
    "overstrain_index",
]


def load_config(config_path: str = "configs/train.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )


def evaluate_predictions(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
    }


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["processed_path"])
    target_col = cfg["data"]["target_col"]

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[target_col]

    # Стратифицированное разбиение сохраняет долю поломок (3.39%) в train и test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    seed = cfg["data"]["random_state"]
    models_cfg = cfg["models"]

    candidates = {
        "logistic_regression": (
            LogisticRegression(
                C=models_cfg["logistic_regression"]["C"],
                max_iter=models_cfg["logistic_regression"]["max_iter"],
                class_weight="balanced",
                random_state=seed,
            ),
            models_cfg["logistic_regression"],
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=models_cfg["random_forest"]["n_estimators"],
                max_depth=models_cfg["random_forest"]["max_depth"],
                class_weight="balanced",
                random_state=seed,
            ),
            models_cfg["random_forest"],
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=models_cfg["gradient_boosting"]["max_iter"],
                learning_rate=models_cfg["gradient_boosting"]["learning_rate"],
                max_depth=models_cfg["gradient_boosting"]["max_depth"],
                class_weight="balanced",
                random_state=seed,
            ),
            models_cfg["gradient_boosting"],
        ),
    }

    best_name = ""
    best_f1 = -1.0
    best_pipeline = None
    all_results = {}

    for name, (estimator, params) in candidates.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("classifier", estimator),
                ]
            )
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate_predictions(y_test, y_pred, y_proba)

            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            all_results[name] = metrics
            print(f"\n[{name}] Метрики на тесте: {metrics}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_name = name
                best_pipeline = pipeline

    model_path = Path(cfg["output"]["model_path"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path)
    summary = {
        "best_model": best_name,
        "best_f1": best_f1,
        "models": all_results,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nПобедитель: {best_name} (F1 = {best_f1})")
    print(f"Модель сохранена в {model_path}, отчёт в {metrics_path}")


if __name__ == "__main__":
    main()
