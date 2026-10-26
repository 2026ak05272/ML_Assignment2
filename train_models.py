"""Train, compare, and persist churn classifiers for the assignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
SOURCE_DATA = (
    ROOT.parent
    / "Sample Data"
    / "Classification"
    / "Telecome Customer Churn"
    / "Telecom_customer_churn.csv"
)
MODEL_DIR = ROOT / "model"
TARGET = "Churn"
TEST_SIZE = 0.20
RANDOM_STATE = 42

MODEL_FACTORIES: dict[str, Any] = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Naive Bayes": lambda: GaussianNB(),
    "Support Vector Machine": lambda: LinearSVC(
        C=0.5, class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000
    ),
}


def build_pipeline(estimator: Any, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("classifier", estimator)])


def main() -> None:
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"Sample dataset not found: {SOURCE_DATA}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE_DATA)
    data.columns = data.columns.str.strip()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
    data[TARGET] = data[TARGET].astype(str).str.strip().str.title()
    data = data[data[TARGET].isin(["Yes", "No"])].copy()
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_data = X_test.copy()
    test_data[TARGET] = y_test.map({0: "No", 1: "Yes"})
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    metrics: dict[str, dict[str, float]] = {}
    saved_models: dict[str, str] = {}

    for model_name, estimator_factory in MODEL_FACTORIES.items():
        pipeline = build_pipeline(estimator_factory(), numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        classifier = pipeline.named_steps["classifier"]
        probabilities = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(classifier, "predict_proba")
            else pipeline.decision_function(X_test)
        )
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "auc": float(roc_auc_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, predictions)),
        }
        filename = model_name.lower().replace("-", "").replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)
        saved_models[model_name] = filename
        print(f"{model_name}: {metrics[model_name]}")

    metadata = {
        "target": TARGET,
        "positive_label": "Yes",
        "negative_label": "No",
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "models": saved_models,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved {len(saved_models)} models and the held-out test data.")


if __name__ == "__main__":
    main()