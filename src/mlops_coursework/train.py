import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
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

from mlops_coursework.features import (
    BASE_FEATURE_COLUMNS,
    ENGINEERED_FEATURE_COLUMNS,
    load_config,
)

CATEGORICAL_COLS = ["machine_type"]
NUMERIC_COLS = [c for c in BASE_FEATURE_COLUMNS if c != "machine_type"] + ENGINEERED_FEATURE_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )


def main() -> None:
    cfg = load_config()["train"]
    df = pd.read_csv(cfg["processed_data_path"])

    x_data = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y_data = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y_data,
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow_tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment_name"])

    models_cfg = cfg["models"]
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=models_cfg["logistic_regression"]["max_iter"],
            C=models_cfg["logistic_regression"]["C"],
            class_weight="balanced",
            random_state=cfg["random_state"],
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=models_cfg["random_forest"]["n_estimators"],
            max_depth=models_cfg["random_forest"]["max_depth"],
            class_weight="balanced",
            random_state=cfg["random_state"],
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=models_cfg["hist_gradient_boosting"]["max_iter"],
            learning_rate=models_cfg["hist_gradient_boosting"]["learning_rate"],
            max_depth=models_cfg["hist_gradient_boosting"]["max_depth"],
            class_weight="balanced",
            random_state=cfg["random_state"],
        ),
    }

    best_f1 = -1.0
    best_pipeline: Pipeline | None = None
    best_name = ""
    all_metrics: dict[str, dict[str, float]] = {}

    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
            pipeline.fit(x_train, y_train)

            y_pred = pipeline.predict(x_test)
            y_prob = pipeline.predict_proba(x_test)[:, 1]

            metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred)), 4),
                "recall": round(float(recall_score(y_test, y_pred)), 4),
                "f1": round(float(f1_score(y_test, y_pred)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
                "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
            }
            all_metrics[name] = metrics

            mlflow.log_params({"model_type": name, **models_cfg[name]})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_pipeline = pipeline
                best_name = name

    model_path = Path(cfg["model_output_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    metrics_path = Path(cfg["metrics_output_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"best_model": best_name, "results": all_metrics}, indent=2),
        encoding="utf-8",
    )
    print(f"Best model: {best_name} (F1={best_f1:.4f}) saved to {model_path}")


if __name__ == "__main__":
    main()
