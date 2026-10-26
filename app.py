"""Interactive evaluation dashboard for the telecom churn classifiers."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST_DATA = ROOT / "test_data.csv"
TARGET = "Churn"

st.set_page_config(page_title="Churn Lab | Model evaluation", page_icon="📡", layout="wide")


@st.cache_data
def load_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Model files are missing. Run train_models.py first.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_path: str, modified_at: int):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def evaluate_cached(model_path: str, modified_at: int, frame: pd.DataFrame):
    """Cache model scores so widget changes don't repeat predictions."""
    model = load_model(model_path, modified_at)
    return score_test_data(model, frame, TARGET)


def score_test_data(model, frame: pd.DataFrame, target_name: str) -> tuple[dict[str, float], list, list, str]:
    labels = frame[target_name].astype(str).str.strip().str.title()
    valid = labels.isin(["Yes", "No"])
    frame = frame.loc[valid].copy()
    y_true = labels.loc[valid].map({"No": 0, "Yes": 1}).astype(int)
    X_test = frame.drop(columns=[target_name])
    predictions = model.predict(X_test)
    classifier = model.named_steps["classifier"]
    probabilities = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(classifier, "predict_proba")
        else model.decision_function(X_test)
    )
    metrics = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "AUC": float(roc_auc_score(y_true, probabilities)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, predictions)),
    }
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["No churn", "Churn"],
        output_dict=False,
        zero_division=0,
    )
    return metrics, y_true.tolist(), predictions.tolist(), report


st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f4f8fc 0%, #eef3f7 55%, #f8f5ef 100%); }
    .block-container { max-width: 1200px; padding-top: 2.3rem; }
    .hero { padding: 1.8rem 2rem; border-radius: 22px; color: #f8fafc;
            background: linear-gradient(110deg, #102a43, #176b70 68%, #d18b47); margin-bottom: 1.2rem; }
    .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -0.04em; }
    .hero p { margin: .55rem 0 0; opacity: .88; }
    div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e1e8ef;
        padding: 1rem; border-radius: 15px; box-shadow: 0 4px 16px #102a430b; }
    </style>
    <div class="hero"><h1>📡 Churn Lab</h1>
    <p>Five classifiers. One held-out test set. Clear, comparable evidence.</p></div>
    """,
    unsafe_allow_html=True,
)

try:
    metadata = load_metadata()
except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

model_names = list(metadata["models"])
with st.sidebar:
    st.header("Evaluation setup")
    selected_model_name = st.selectbox("Choose a classifier", model_names)
    uploaded_file = st.file_uploader("Upload held-out test data (CSV)", type=["csv"])
    st.caption("CSV must contain the saved model's input columns and a Churn target (Yes/No).")
    st.divider()
    st.caption(f"Training rows: {metadata['train_rows']:,} · Test rows: {metadata['test_rows']:,}")

try:
    test_data = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DEFAULT_TEST_DATA)
except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

test_data.columns = test_data.columns.str.strip()
missing_columns = sorted(set(metadata["features"] + [TARGET]) - set(test_data.columns))
if missing_columns:
    st.error("The test CSV is missing required columns: " + ", ".join(missing_columns))
    st.stop()
if test_data.empty:
    st.error("The selected test CSV contains no rows.")
    st.stop()

test_data = test_data[metadata["features"] + [TARGET]].copy()
for column in metadata["numeric_features"]:
    test_data[column] = pd.to_numeric(test_data[column], errors="coerce")

selected_model_path = str(MODEL_DIR / metadata["models"][selected_model_name])
selected_model_mtime = Path(selected_model_path).stat().st_mtime_ns

try:
    scores, y_true, predictions, report = evaluate_cached(
        selected_model_path, selected_model_mtime, test_data
    )
except (ValueError, KeyError) as exc:
    st.error(f"Unable to evaluate this file. Check the feature values and Yes/No target labels. Details: {exc}")
    st.stop()

st.subheader("Selected model")
st.write(f"**{selected_model_name}** · evaluated on {len(y_true):,} labeled rows")
metric_columns = st.columns(6)
for column, (name, value) in zip(metric_columns, scores.items()):
    column.metric(name, f"{value:.3f}")

left, right = st.columns([0.9, 1.1], gap="large")
with left:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    matrix_frame = pd.DataFrame(matrix, index=["Actual: No", "Actual: Yes"], columns=["Predicted: No", "Predicted: Yes"])
    st.dataframe(matrix_frame, width="stretch")
    st.caption("Rows are actual churn labels; columns are model predictions.")

with right:
    st.subheader("Classification report")
    st.code(report, language="text")

st.divider()
st.subheader("All-model comparison")
comparison = pd.DataFrame(
    [
        {
            "Model": name,
            "Accuracy": values["accuracy"],
            "AUC": values["auc"],
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1"],
            "MCC": values["mcc"],
        }
        for name, values in metadata["metrics"].items()
    ]
).set_index("Model")

if uploaded_file is None:
    st.caption("Precomputed on the bundled held-out test set; comparison loads instantly.")
    st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
else:
    comparison_key = hashlib.sha256(test_data.to_csv(index=False).encode("utf-8")).hexdigest()
    if st.session_state.get("comparison_key") != comparison_key:
        st.info("Compare all five models on the uploaded CSV when you are ready.")
        if st.button("Compare all models", type="primary"):
            comparison_rows = []
            progress = st.progress(0, text="Evaluating models…")
            for index, name in enumerate(model_names, start=1):
                model_path = str(MODEL_DIR / metadata["models"][name])
                model_mtime = Path(model_path).stat().st_mtime_ns
                scores_for_model, _, _, _ = evaluate_cached(model_path, model_mtime, test_data)
                comparison_rows.append({"Model": name, **scores_for_model})
                progress.progress(index / len(model_names), text=f"Evaluated {name}")
            st.session_state["comparison_key"] = comparison_key
            st.session_state["comparison_rows"] = comparison_rows
            st.rerun()
    if st.session_state.get("comparison_key") == comparison_key:
        comparison = pd.DataFrame(st.session_state["comparison_rows"]).set_index("Model")
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")

st.subheader("Model performance observations")
st.caption(
    "Observations use the bundled held-out test results, or the uploaded test results "
    "after selecting Compare all models."
)
metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metric_leaders = {
    metric: comparison[metric].idxmax()
    for metric in metric_columns
}
observation_rows = []
for model_name, row in comparison.iterrows():
    leading_metrics = [metric for metric, leader in metric_leaders.items() if leader == model_name]
    if leading_metrics:
        observation = "Highest " + ", ".join(leading_metrics) + ". "
    else:
        observation = ""
    observation += (
        f"Accuracy {row['Accuracy']:.3f}, AUC {row['AUC']:.3f}, "
        f"precision {row['Precision']:.3f}, recall {row['Recall']:.3f}, "
        f"F1 {row['F1']:.3f}, MCC {row['MCC']:.3f}."
    )
    observation_rows.append({"ML Model Name": model_name, "Observation about model performance": observation})

balanced_winner = comparison.sort_values(["F1", "MCC"], ascending=False).index[0]
auc_winner = metric_leaders["AUC"]
winner_note = (
    f"{balanced_winner} is the overall balanced winner by F1 "
    f"({comparison.loc[balanced_winner, 'F1']:.3f}) and MCC "
    f"({comparison.loc[balanced_winner, 'MCC']:.3f}). "
)
if auc_winner != balanced_winner:
    winner_note += f"{auc_winner} has the highest AUC ({comparison.loc[auc_winner, 'AUC']:.3f})."
else:
    winner_note += f"It also has the highest AUC ({comparison.loc[auc_winner, 'AUC']:.3f})."
observation_rows.append(
    {"ML Model Name": "Overall, Winner for your dataset?", "Observation about model performance": winner_note}
)
observations = pd.DataFrame(observation_rows).set_index("ML Model Name")
st.dataframe(observations, width="stretch")

st.caption("Models are loaded from disk and never retrained on uploaded test data. Repeated evaluations are cached.")