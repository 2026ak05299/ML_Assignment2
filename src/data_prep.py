from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "obesity_levels.csv"
TARGET_COLUMN = "NObeyesdad"

MIN_FEATURES = 12
MIN_ROWS = 500

# The target classes are defined by BMI = Weight / Height^2, so these two columns leak the label.
LEAKY_FEATURES = ["Height", "Weight"]


def load_obesity_data(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path).drop(columns=LEAKY_FEATURES)


def split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = frame.drop(columns=[TARGET_COLUMN])
    categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = features.select_dtypes(include="number").columns.tolist()
    return categorical, numeric


def describe_dataset(frame: pd.DataFrame) -> None:
    n_rows, n_cols = frame.shape
    n_features = n_cols - 1
    categorical, numeric = split_feature_types(frame)

    print(f"Shape: {n_rows} rows x {n_cols} columns ({n_features} features + target)")
    print(f"Meets minimum size: features {n_features} >= {MIN_FEATURES} -> {n_features >= MIN_FEATURES}, "
          f"rows {n_rows} >= {MIN_ROWS} -> {n_rows >= MIN_ROWS}")

    print(f"\nCategorical features ({len(categorical)}):")
    for col in categorical:
        print(f"  {col:32s} {sorted(frame[col].unique())}")

    print(f"\nNumeric features ({len(numeric)}):")
    print(frame[numeric].describe().T[["mean", "std", "min", "max"]].round(2).to_string())

    missing = frame.isna().sum()
    print(f"\nMissing values: {int(missing.sum())} total")
    if missing.any():
        print(missing[missing > 0].to_string())

    print(f"\nDuplicate rows: {frame.duplicated().sum()}")

    print(f"\nTarget '{TARGET_COLUMN}' distribution:")
    counts = frame[TARGET_COLUMN].value_counts()
    share = (counts / n_rows * 100).round(1)
    print(pd.DataFrame({"count": counts, "percent": share}).to_string())


if __name__ == "__main__":
    describe_dataset(load_obesity_data())
