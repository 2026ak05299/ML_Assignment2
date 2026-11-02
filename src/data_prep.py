from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "obesity_levels.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "train_data.csv"
TEST_PATH = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "NObeyesdad"

TEST_SIZE = 0.20
SPLIT_SEED = 2026

FREQUENCY_LEVELS = ["no", "Sometimes", "Frequently", "Always"]
ORDINAL_FEATURES = ["CAEC", "CALC"]
NOMINAL_FEATURES = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "MTRANS"]
NUMERIC_FEATURES = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

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


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("frequency", OrdinalEncoder(categories=[FREQUENCY_LEVELS] * len(ORDINAL_FEATURES),
                                         handle_unknown="use_encoded_value", unknown_value=-1),
             ORDINAL_FEATURES),
            ("nominal", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
             NOMINAL_FEATURES),
        ],
        remainder="drop",
    )


def make_train_test_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deduplicated = frame.drop_duplicates().reset_index(drop=True)
    train_frame, test_frame = train_test_split(
        deduplicated,
        test_size=TEST_SIZE,
        stratify=deduplicated[TARGET_COLUMN],
        random_state=SPLIT_SEED,
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def save_splits(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    train_frame.to_csv(TRAIN_PATH, index=False)
    test_frame.to_csv(TEST_PATH, index=False)


def summarise_split(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    print(f"\nTrain rows: {len(train_frame)} -> {TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Test rows:  {len(test_frame)} -> {TEST_PATH.relative_to(PROJECT_ROOT)}")
    share = pd.DataFrame({
        "train %": (train_frame[TARGET_COLUMN].value_counts(normalize=True) * 100).round(1),
        "test %": (test_frame[TARGET_COLUMN].value_counts(normalize=True) * 100).round(1),
    })
    print(share.to_string())

    encoded = build_preprocessor().fit(train_frame.drop(columns=[TARGET_COLUMN]))
    print(f"\nEncoded feature count after preprocessing: {len(encoded.get_feature_names_out())}")


if __name__ == "__main__":
    obesity = load_obesity_data()
    describe_dataset(obesity)
    train_set, test_set = make_train_test_split(obesity)
    save_splits(train_set, test_set)
    summarise_split(train_set, test_set)
