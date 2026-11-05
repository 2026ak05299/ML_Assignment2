import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_prep import (  # noqa: E402
    LEAKY_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
    TEST_PATH,
)
from evaluate import METRIC_COLUMNS, MODEL_DIR, MODEL_DISPLAY_NAMES, score_model  # noqa: E402

REQUIRED_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

OBESITY_LEVELS = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
SHORT_LEVEL_NAMES = {
    "Insufficient_Weight": "Insufficient",
    "Normal_Weight": "Normal",
    "Overweight_Level_I": "Overweight I",
    "Overweight_Level_II": "Overweight II",
    "Obesity_Type_I": "Obesity I",
    "Obesity_Type_II": "Obesity II",
    "Obesity_Type_III": "Obesity III",
}
FEATURE_GLOSSARY = {
    "FAVC": "Frequently eats high-calorie food",
    "FCVC": "Vegetable consumption frequency (1-3)",
    "NCP": "Number of main meals per day (1-4)",
    "CAEC": "Eating between meals",
    "CH2O": "Daily water intake (1-3)",
    "SCC": "Monitors calorie intake",
    "FAF": "Physical activity frequency (0-3)",
    "TUE": "Time on technology devices (0-2)",
    "CALC": "Alcohol consumption",
    "MTRANS": "Usual mode of transport",
}

st.set_page_config(page_title="Obesity Level Classifier Lab", page_icon="🥗", layout="wide")


@st.cache_resource
def load_model(file_stem: str):
    return joblib.load(MODEL_DIR / f"{file_stem}.joblib")


@st.cache_data
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def validate_upload(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    errors, notes = [], []

    missing = [col for col in REQUIRED_FEATURES + [TARGET_COLUMN] if col not in frame.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return None, errors, notes

    non_numeric = [col for col in NUMERIC_FEATURES if not pd.api.types.is_numeric_dtype(frame[col])]
    if non_numeric:
        errors.append(f"These columns must be numeric: {', '.join(non_numeric)}")

    unknown_labels = sorted(set(frame[TARGET_COLUMN].dropna()) - set(OBESITY_LEVELS))
    if unknown_labels:
        errors.append(f"Unknown values in '{TARGET_COLUMN}': {', '.join(map(str, unknown_labels))}")
    if errors:
        return None, errors, notes

    ignored = [col for col in frame.columns if col not in REQUIRED_FEATURES + [TARGET_COLUMN]]
    if ignored:
        leak_note = " (Height/Weight are excluded by design)" if set(ignored) & set(LEAKY_FEATURES) else ""
        notes.append(f"Ignoring extra columns: {', '.join(ignored)}{leak_note}")

    cleaned = frame[REQUIRED_FEATURES + [TARGET_COLUMN]].dropna()
    if len(cleaned) < len(frame):
        notes.append(f"Dropped {len(frame) - len(cleaned)} rows with missing values")
    if cleaned.empty:
        errors.append("No complete rows left to evaluate")
        return None, errors, notes
    return cleaned.reset_index(drop=True), errors, notes


def format_metric(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.3f}"


def plot_confusion_matrix(labels: pd.Series, predictions, model_name: str):
    levels = [lvl for lvl in OBESITY_LEVELS if lvl in set(labels) | set(predictions)]
    matrix = confusion_matrix(labels, predictions, labels=levels)
    short = [SHORT_LEVEL_NAMES[lvl] for lvl in levels]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="crest", cbar=False,
                xticklabels=short, yticklabels=short, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("Actual level")
    ax.set_title(f"{model_name}: actual vs predicted")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def report_table(labels: pd.Series, predictions) -> pd.DataFrame:
    levels = [lvl for lvl in OBESITY_LEVELS if lvl in set(labels)]
    report = classification_report(labels, predictions, labels=levels, output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T.drop(index="accuracy", errors="ignore")
    table = table.rename(index=SHORT_LEVEL_NAMES)
    table["support"] = table["support"].astype(int)
    return table.rename(columns=str.title)


@st.cache_data
def compare_all_models(frame: pd.DataFrame) -> pd.DataFrame:
    features, labels = frame[REQUIRED_FEATURES], frame[TARGET_COLUMN]
    rows = []
    for file_stem, display_name in MODEL_DISPLAY_NAMES.items():
        rows.append({"Model": display_name, **score_model(load_model(file_stem), features, labels)})
    return pd.DataFrame(rows)


def plot_comparison(results: pd.DataFrame):
    long_form = results.melt(id_vars="Model", value_vars=METRIC_COLUMNS, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sns.barplot(data=long_form, x="Metric", y="Score", hue="Model", palette="crest", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()
    return fig


# ---------- Sidebar ----------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help="Upload held-out test data only. The models are pre-trained and are not refit.")
    use_bundled = st.checkbox("Use the bundled test_data.csv instead", value=uploaded is None)
    st.download_button("Download sample test_data.csv", data=TEST_PATH.read_bytes(),
                       file_name="test_data.csv", mime="text/csv")

    st.header("2. Classifier")
    chosen_stem = st.selectbox("Model", options=list(MODEL_DISPLAY_NAMES),
                               format_func=MODEL_DISPLAY_NAMES.get, index=4)

    with st.expander("Expected columns"):
        st.markdown(f"**Target:** `{TARGET_COLUMN}`")
        st.markdown("**Features:** " + ", ".join(f"`{col}`" for col in REQUIRED_FEATURES))
        st.caption("Height and Weight are deliberately not used: the obesity levels are defined by BMI, "
                   "so those two columns would give away the answer.")

# ---------- Header ----------
st.title("🥗 Obesity Level Classifier Lab")
st.markdown(
    "Can **eating habits and lifestyle alone** predict someone's obesity level? "
    "Five pre-trained classifiers are scored on your test data across the seven levels, "
    "from *Insufficient Weight* to *Obesity Type III*."
)

source = uploaded if uploaded is not None else (TEST_PATH if use_bundled else None)
if source is None:
    st.info("Upload a test CSV in the sidebar, or tick **Use the bundled test_data.csv** to get started.")
    st.stop()

raw_frame = read_csv(source)
test_frame, problems, notes = validate_upload(raw_frame)
for problem in problems:
    st.error(problem)
if test_frame is None:
    st.stop()
for note in notes:
    st.warning(note)

source_name = uploaded.name if uploaded is not None else "bundled test_data.csv"
st.caption(f"Evaluating **{len(test_frame)} rows** from *{source_name}* · "
           "Precision / Recall / F1 are macro averages · AUC is one-vs-rest macro")

features, labels = test_frame[REQUIRED_FEATURES], test_frame[TARGET_COLUMN]
model = load_model(chosen_stem)
model_name = MODEL_DISPLAY_NAMES[chosen_stem]
predictions = model.predict(features)
scores = score_model(model, features, labels)

single_tab, compare_tab, data_tab = st.tabs(["🔍 Selected model", "📊 Compare all five", "🗂️ Test data"])

with single_tab:
    st.subheader(model_name)
    for column, metric in zip(st.columns(len(METRIC_COLUMNS)), METRIC_COLUMNS):
        column.metric(metric, format_metric(scores[metric]), border=True)

    matrix_col, report_col = st.columns([1.1, 1])
    with matrix_col:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion_matrix(labels, predictions, model_name))
        st.caption("Levels run from lightest to heaviest, so errors next to the diagonal "
                   "are near-misses between neighbouring levels.")
    with report_col:
        st.markdown("**Classification report**")
        st.dataframe(report_table(labels, predictions).style.format(precision=3), height=420)

with compare_tab:
    comparison = compare_all_models(test_frame)
    leader = comparison.loc[comparison["F1"].idxmax(), "Model"]
    st.success(f"Best macro F1 on this data: **{leader}**")
    st.dataframe(
        comparison.set_index("Model").style.format(precision=4).highlight_max(color="#BFE3DC"),
        width="stretch",
    )
    st.pyplot(plot_comparison(comparison))

with data_tab:
    distribution = (labels.value_counts().reindex(OBESITY_LEVELS).dropna().astype(int)
                    .rename(index=SHORT_LEVEL_NAMES).rename("rows"))
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Rows per obesity level**")
        st.bar_chart(distribution, color="#0F766E")
    with right:
        st.markdown("**Preview with predictions**")
        preview = test_frame.assign(Predicted=predictions)
        st.dataframe(preview.head(50), height=360)
    with st.expander("What do the feature codes mean?"):
        st.table(pd.DataFrame.from_dict(FEATURE_GLOSSARY, orient="index", columns=["Meaning"]))
