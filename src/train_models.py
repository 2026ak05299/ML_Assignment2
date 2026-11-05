import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_prep import PROJECT_ROOT, SPLIT_SEED, TARGET_COLUMN, TRAIN_PATH, build_preprocessor

# SVC(probability=True) is deprecated in scikit-learn 1.9 but still supported until 1.11;
# the env var silences the same warning inside GridSearchCV's worker processes.
SVC_PROBABILITY_WARNING = "ignore:The `probability` parameter was deprecated:FutureWarning"
os.environ["PYTHONWARNINGS"] = SVC_PROBABILITY_WARNING
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated", category=FutureWarning)

MODEL_DIR = PROJECT_ROOT / "model"
TUNING_SUMMARY_PATH = MODEL_DIR / "tuning_summary.csv"

CV_FOLDS = 5
TUNING_METRIC = "f1_macro"

CLASSIFIER_SEARCH_SPACE = {
    "logistic_regression": (
        LogisticRegression(max_iter=3000),
        {"C": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=SPLIT_SEED),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [6, 10, 15, 20, None],
            "min_samples_leaf": [1, 3, 5],
        },
    ),
    "knn": (
        KNeighborsClassifier(),
        {"n_neighbors": [1, 3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
    ),
    "naive_bayes": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 0.3, 1.0]},
    ),
    "svm": (
        SVC(kernel="rbf", probability=True, random_state=SPLIT_SEED),
        {"C": [1.0, 10.0, 30.0, 100.0], "gamma": ["scale", 0.05, 0.2, 0.5]},
    ),
}


def load_training_data(path: Path = TRAIN_PATH) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def tune_classifier(name: str, features: pd.DataFrame, labels: pd.Series) -> GridSearchCV:
    estimator, grid = CLASSIFIER_SEARCH_SPACE[name]
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("classifier", estimator)])
    search = GridSearchCV(
        pipeline,
        param_grid={f"classifier__{param}": values for param, values in grid.items()},
        scoring=TUNING_METRIC,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SPLIT_SEED),
        n_jobs=-1,
    )
    return search.fit(features, labels)


def train_all_models() -> pd.DataFrame:
    MODEL_DIR.mkdir(exist_ok=True)
    features, labels = load_training_data()
    summary_rows = []

    for name in CLASSIFIER_SEARCH_SPACE:
        search = tune_classifier(name, features, labels)
        joblib.dump(search.best_estimator_, MODEL_DIR / f"{name}.joblib")

        best_params = {key.removeprefix("classifier__"): value for key, value in search.best_params_.items()}
        summary_rows.append({
            "model": name,
            "best_params": best_params,
            f"cv_{TUNING_METRIC}": round(search.best_score_, 4),
        })
        print(f"{name:20s} CV {TUNING_METRIC} = {search.best_score_:.4f}  best = {best_params}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TUNING_SUMMARY_PATH, index=False)
    return summary


if __name__ == "__main__":
    train_all_models()
    print(f"\nSaved models and tuning summary to {MODEL_DIR.relative_to(PROJECT_ROOT)}")
