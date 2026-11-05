from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_prep import PROJECT_ROOT, TARGET_COLUMN, TEST_PATH

MODEL_DIR = PROJECT_ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.csv"

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def load_test_data(path: Path = TEST_PATH) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def multiclass_auc(labels: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    """One-vs-rest macro AUC over the classes present in `labels` (NaN if fewer than two are present)."""
    present = [i for i, cls in enumerate(classes) if cls in set(labels)]
    if len(present) < 2:
        return float("nan")
    if len(classes) == 2:
        return roc_auc_score(labels == classes[1], probabilities[:, 1])
    per_class = [roc_auc_score(labels == classes[i], probabilities[:, i]) for i in present]
    return float(np.mean(per_class))


def score_model(model, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": multiclass_auc(labels, probabilities, model.classes_),
        "Precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "Recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "F1": f1_score(labels, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def evaluate_saved_models(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows = []
    for file_stem, display_name in MODEL_DISPLAY_NAMES.items():
        model = joblib.load(MODEL_DIR / f"{file_stem}.joblib")
        rows.append({"ML Model Name": display_name, **score_model(model, features, labels)})
    return pd.DataFrame(rows)


def to_markdown_table(results: pd.DataFrame) -> str:
    header = ["ML Model Name", *METRIC_COLUMNS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in results.iterrows():
        values = [f"{row[col]:.4f}" for col in METRIC_COLUMNS]
        lines.append(f"| {row['ML Model Name']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    test_features, test_labels = load_test_data()
    metrics = evaluate_saved_models(test_features, test_labels)
    metrics.round(4).to_csv(METRICS_PATH, index=False)

    print(f"Test rows: {len(test_labels)} | Precision/Recall/F1: macro average | AUC: one-vs-rest macro\n")
    print(to_markdown_table(metrics))
    print(f"\nSaved metrics to {METRICS_PATH.relative_to(PROJECT_ROOT)}")
