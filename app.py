"""Interactive, upload-first dashboard for telecom churn model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
SAMPLE_PATH = ROOT / "test_data.csv"
TARGET = "Churn"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 20_000

st.set_page_config(
    page_title="Churn Atlas | Telecom analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any]:
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("The saved model metadata is missing from the model folder.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, modified_at: int):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def evaluate_all_models(data: pd.DataFrame, model_entries: tuple[tuple[str, str, int], ...]):
    """Evaluate every saved pipeline on the same uploaded, labeled rows."""
    labels = data[TARGET].astype(str).str.strip().str.casefold()
    y_true = labels.map({"no": 0, "yes": 1}).astype(int)
    X = data.drop(columns=[TARGET])
    results: dict[str, dict[str, Any]] = {}

    for model_name, model_path, modified_at in model_entries:
        model = load_model(model_path, modified_at)
        predictions = model.predict(X)
        classifier = model.named_steps["classifier"]
        positive_scores = (
            model.predict_proba(X)[:, 1]
            if hasattr(classifier, "predict_proba")
            else model.decision_function(X)
        )
        results[model_name] = {
            "metrics": {
                "Accuracy": float(accuracy_score(y_true, predictions)),
                "AUC": float(roc_auc_score(y_true, positive_scores)),
                "Precision": float(precision_score(y_true, predictions, zero_division=0)),
                "Recall": float(recall_score(y_true, predictions, zero_division=0)),
                "F1": float(f1_score(y_true, predictions, zero_division=0)),
                "MCC": float(matthews_corrcoef(y_true, predictions)),
            },
            "predictions": predictions.tolist(),
            "report": classification_report(
                y_true,
                predictions,
                labels=[0, 1],
                target_names=["No churn", "Churn"],
                zero_division=0,
                output_dict=True,
            ),
        }
        results[model_name]["report"]["accuracy"] = float(accuracy_score(y_true, predictions))
    return y_true.tolist(), results


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#192b35; --muted:#60717a; --paper:#e9efeb; --line:#d2ded7; --green:#bdd867; --deep:#173b36; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stSidebar"] { background:#e9eee5; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
    .block-container { max-width:1420px; padding-top:2rem; padding-bottom:4rem; }
    .brand { display:flex; align-items:center; gap:12px; margin:0 0 1.8rem; }
    .brand-mark { width:42px; height:42px; display:grid; place-items:center; background:var(--green); color:var(--deep); border-radius:14px; font-size:22px; font-weight:800; }
    .brand-word { font-size:1.08rem; line-height:1.05; font-weight:800; letter-spacing:-.04em; }
    .brand-sub { color:var(--muted); font:500 .62rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.13em; margin-top:5px; }
    .eyebrow { color:#587067; font:500 .68rem 'DM Mono',monospace; letter-spacing:.17em; text-transform:uppercase; }
    .hero { border-radius:26px; padding:2.1rem 2.25rem; color:#f2f4e9; background:#173b36; position:relative; overflow:hidden; margin:.6rem 0 1.2rem; }
    .hero:after { content:'◈'; position:absolute; right:40px; top:-54px; color:#bdd867; opacity:.92; font-size:205px; line-height:1; transform:rotate(12deg); }
    .hero h1 { font-size:clamp(2rem,4vw,3.55rem); letter-spacing:-.075em; line-height:1.03; margin:.55rem 0 .7rem; max-width:720px; }
    .hero p { color:#d5e0d4; max-width:680px; font-size:.98rem; line-height:1.65; margin:0; }
    .hero .eyebrow { color:#bdd867; }
    .section-title { font-size:1.3rem; font-weight:800; letter-spacing:-.04em; margin:.35rem 0 .25rem; }
    .muted { color:var(--muted); font-size:.9rem; line-height:1.55; }
    .data-card { border:1px solid var(--line); background:#f4f7f3; border-radius:18px; padding:1.1rem 1.2rem; min-height:112px; }
    .data-label { color:var(--muted); font:500 .66rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; }
    .data-value { color:var(--ink); font-size:1.9rem; font-weight:800; letter-spacing:-.06em; margin-top:.45rem; }
    .data-note { color:var(--muted); font-size:.76rem; margin-top:.1rem; }
    .step-tag { display:inline-block; color:var(--deep); background:#e3ecc6; padding:.4rem .7rem; border-radius:30px; font:500 .68rem 'DM Mono',monospace; letter-spacing:.06em; }
    .upload-hint { border:1px dashed #98a995; border-radius:18px; background:#eff2e9; padding:1rem 1.2rem; margin:.5rem 0; }
    div[data-testid="stMetric"] { background:#f1f5f1; border:1px solid var(--line); border-radius:16px; padding:1rem; }
    div[data-testid="stMetric"] label { color:var(--muted); }
    .stButton > button[kind="primary"] { background:var(--deep); border-color:var(--deep); border-radius:12px; font-weight:700; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    .diagnostic-card { background:#e2ebe6; border:1px solid #cbd9d1; border-radius:20px; padding:1.15rem; min-height:300px; }
    .diagnostic-kicker { color:#587067; font:500 .65rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.12em; margin-bottom:.8rem; }
    .heatmap { width:100%; border-collapse:separate; border-spacing:6px; table-layout:fixed; }
    .heatmap th { color:#52646d; font-size:.72rem; font-weight:700; padding:.45rem .25rem; text-align:center; }
    .heatmap th.row-head { text-align:left; width:24%; }
    .heatmap td { text-align:center; border-radius:13px; padding:.75rem .35rem; color:#19332f; }
    .heatmap td.tn { background:#c5ded1; }
    .heatmap td.fp { background:#f1dfb5; }
    .heatmap td.fn { background:#eac8bd; }
    .heatmap td.tp { background:#a8cfbb; }
    .heat-value { display:block; font-size:1.45rem; line-height:1.2; font-weight:800; letter-spacing:-.04em; }
    .heat-share { display:block; font:500 .62rem 'DM Mono',monospace; opacity:.72; margin-top:.22rem; }
    .heat-key { display:flex; flex-wrap:wrap; gap:.45rem 1rem; color:#51645d; font-size:.69rem; margin:.65rem .15rem 0; }
    .key-item { display:inline-flex; align-items:center; gap:.35rem; }
    .key-swatch { display:inline-block; width:10px; height:10px; border-radius:3px; }
    .key-swatch.good { background:#a8cfbb; }
    .key-swatch.warn { background:#f1dfb5; }
    .key-swatch.miss { background:#eac8bd; }
    .report-table { width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border-radius:13px; background:#edf2f3; font-size:.78rem; }
    .report-table th { background:#d5e0e2; color:#334b52; text-align:left; font-size:.67rem; text-transform:uppercase; letter-spacing:.06em; padding:.7rem .55rem; }
    .report-table td { color:#283b43; padding:.72rem .55rem; border-top:1px solid #dce5e6; vertical-align:middle; }
    .report-table tr:nth-child(even) td { background:#e7eef0; }
    .class-name { font-weight:800; white-space:nowrap; }
    .score-cell { min-width:70px; }
    .score-value { display:flex; justify-content:space-between; font:500 .68rem 'DM Mono',monospace; margin-bottom:.28rem; }
    .score-track { height:5px; background:#d4dfe0; border-radius:10px; overflow:hidden; }
    .score-fill { display:block; height:100%; border-radius:10px; background:#648b78; }
    .report-support { color:#687b80; font:500 .7rem 'DM Mono',monospace; }
    .footnote { color:var(--muted); font-size:.76rem; border-top:1px solid var(--line); padding-top:1rem; margin-top:2.2rem; }
    </style>
    <div class="brand"><div class="brand-mark">◈</div><div><div class="brand-word">CHURN ATLAS</div><div class="brand-sub">Customer intelligence / ML lab</div></div></div>
    """,
    unsafe_allow_html=True,
)

try:
    metadata = load_metadata()
except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
    st.error(f"Could not load the saved model information: {exc}")
    st.stop()

feature_count = len(metadata["features"])
model_names = list(metadata["models"])

with st.sidebar:
    st.markdown('<div class="eyebrow">01 / Evaluation data</div>', unsafe_allow_html=True)
    st.markdown("### Bring a labeled test set")
    st.caption("Upload a test CSV to see how each saved classifier performs on the same customer records.")
    sample_bytes = SAMPLE_PATH.read_bytes() if SAMPLE_PATH.exists() else b""
    if sample_bytes:
        st.download_button(
            "↓  Download sample test CSV",
            data=sample_bytes,
            file_name="telecom_churn_test_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )
    uploaded_file = st.file_uploader(
        "Upload test data",
        type=["csv"],
        help="Include every required feature column and a Churn target with Yes/No labels. Maximum 5 MB and 20,000 rows.",
    )
    st.caption("CSV · up to 5 MB · 20,000 rows · evaluation only")
    with st.expander("Required CSV format"):
        st.write("Your CSV must contain all model input columns plus `Churn`.")
        st.code(", ".join(metadata["features"] + [TARGET]), language="text")
    st.divider()
    st.markdown("**A fair comparison**")
    st.caption("All five saved models evaluate the exact same uploaded rows. The upload is never used for training.")
    st.markdown(f"**Target:** `{TARGET}` · **Models:** {len(model_names)}")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine learning / binary classification</div>
      <h1>Know which customers<br>might be ready to leave.</h1>
      <p>Explore the telecom churn dataset, then upload labeled test data to compare five trained classifiers—without retraining on your file.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<span class="step-tag">DATASET OVERVIEW</span>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Start with the data, then run the models</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">This workspace uses the Telco Customer Churn dataset. The trained pipelines, including preprocessing, are already saved; upload a labeled test CSV when you are ready to evaluate them.</div>',
    unsafe_allow_html=True,
)

card_columns = st.columns(4, gap="medium")
card_data = [
    ("CUSTOMER RECORDS", f"{metadata['train_rows'] + metadata['test_rows']:,}", "source dataset"),
    ("INPUT FEATURES", str(feature_count), "demographic, service & billing"),
    ("SAVED CLASSIFIERS", str(len(model_names)), "ready for evaluation"),
    ("TEST ROWS", f"{metadata['test_rows']:,}", "reserved holdout · 20%"),
]
for column, (label, value, note) in zip(card_columns, card_data):
    with column:
        st.markdown(
            f'<div class="data-card"><div class="data-label">{label}</div><div class="data-value">{value}</div><div class="data-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
intro_left, intro_right = st.columns([1.25, 0.75], gap="large")
with intro_left:
    st.markdown('<div class="section-title">Dataset profile</div>', unsafe_allow_html=True)
    st.markdown(
        "The response variable is **Churn**. A Yes label represents a customer who left; No represents a customer who stayed. The dataset has more non-churn than churn cases, so compare recall and F1 alongside accuracy.",
        unsafe_allow_html=False,
    )
    profile_columns = st.columns(3)
    profile_columns[0].metric("Churned customers", "1,869", "26.5% of source data")
    profile_columns[1].metric("Stayed customers", "5,174", "73.5% of source data")
    profile_columns[2].metric("Missing TotalCharges", "11", "imputed by saved pipeline")
with intro_right:
    st.markdown('<div class="section-title">What is included?</div>', unsafe_allow_html=True)
    st.markdown(
        "- Five pre-trained classifiers\n- Six standard classification metrics\n- Confusion matrix and per-class report\n- Held-out sample CSV for a quick start\n- No model fitting on uploaded test data",
        unsafe_allow_html=False,
    )
    st.link_button("View the Kaggle dataset ↗", "https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn")

st.markdown('<div class="upload-hint"><strong>Ready when you are.</strong> Use the sample download in the left panel, then upload it there. Results will appear below after the CSV passes validation.</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown('<span class="step-tag">02 / RESULTS</span>', unsafe_allow_html=True)
    st.markdown("### Upload your test data to begin")
    st.caption("Your results will appear here after a valid CSV upload: all five models, six metrics, a comparison chart, confusion matrix, and classification report.")
    st.info("No CSV uploaded yet. Download the sample from the sidebar or choose your own labeled test file.")
    st.markdown(
        '<div class="footnote">CHURN ATLAS · ML ASSIGNMENT 2 · EVALUATION WITH SAVED MODELS · NO RETRAINING</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if uploaded_file.size > MAX_UPLOAD_BYTES:
    st.error("This CSV exceeds the 5 MB upload limit. Please use a smaller labeled test file.")
    st.stop()

try:
    uploaded_file.seek(0)
    test_data = pd.read_csv(uploaded_file)
except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
    st.error(f"Could not read this CSV. Check that it is a valid, comma-separated text file. Details: {exc}")
    st.stop()

test_data.columns = test_data.columns.astype(str).str.strip()
missing_columns = sorted(set(metadata["features"] + [TARGET]) - set(test_data.columns))
if missing_columns:
    st.error("The CSV is missing required columns: " + ", ".join(missing_columns))
    st.caption("Download the sample CSV to see the expected column names and order.")
    st.stop()
if test_data.empty:
    st.error("The uploaded CSV contains no data rows.")
    st.stop()
if len(test_data) > MAX_UPLOAD_ROWS:
    st.error(f"This file has {len(test_data):,} rows. The current limit is {MAX_UPLOAD_ROWS:,} rows to keep comparisons responsive.")
    st.stop()

test_data = test_data[metadata["features"] + [TARGET]].copy()
for column in metadata["numeric_features"]:
    test_data[column] = pd.to_numeric(test_data[column], errors="coerce")
labels = test_data[TARGET].astype(str).str.strip().str.casefold()
valid_labels = labels.isin(["yes", "no"])
if not valid_labels.all():
    bad_count = int((~valid_labels).sum())
    st.error(f"The Churn column contains {bad_count} blank or unsupported labels. Use only Yes or No for every row.")
    st.stop()
if labels.nunique() != 2:
    st.error("The uploaded test file must contain both Churn classes (Yes and No) so AUC can be calculated.")
    st.stop()

model_entries = tuple(
    (
        name,
        str(MODEL_DIR / metadata["models"][name]),
        (MODEL_DIR / metadata["models"][name]).stat().st_mtime_ns,
    )
    for name in model_names
)

st.markdown('<span class="step-tag">02 / RESULTS</span>', unsafe_allow_html=True)
st.markdown("### Test file accepted · running the comparison")
st.caption(f"{len(test_data):,} rows · {feature_count} features · target: {TARGET} · same rows for all models")
try:
    with st.spinner("Evaluating the five saved models on your test rows…"):
        y_true, model_results = evaluate_all_models(test_data, model_entries)
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Evaluation could not finish. Check that the input values match the expected feature types. Details: {exc}")
    st.stop()

comparison = pd.DataFrame(
    {model_name: result["metrics"] for model_name, result in model_results.items()}
).T
comparison.index.name = "Model"

st.markdown('<div class="section-title">Five-model scorecard</div>', unsafe_allow_html=True)
summary_metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metric_columns = st.columns(len(summary_metrics), gap="small")
for column, metric_name in zip(metric_columns, summary_metrics):
    best_name = comparison[metric_name].idxmax()
    column.metric(metric_name, f"{comparison.loc[best_name, metric_name]:.3f}", f"Best: {best_name}")

chart_column, table_column = st.columns([0.85, 1.15], gap="large")
with chart_column:
    st.markdown("#### Compare the models")
    chart_metric = st.selectbox(
        "Metric to visualize",
        summary_metrics,
        index=summary_metrics.index("F1"),
        key="chart_metric",
    )
    chart_data = comparison[[chart_metric]].sort_values(chart_metric, ascending=True)
    st.bar_chart(chart_data, horizontal=True, color="#527c58")
with table_column:
    st.markdown("#### Full evaluation table")
    st.dataframe(comparison.style.format("{:.3f}"), width="stretch")

best_f1_model = comparison["F1"].idxmax()
best_mcc_model = comparison["MCC"].idxmax()
best_auc_model = comparison["AUC"].idxmax()
if best_f1_model == best_mcc_model:
    st.success(
        f"Balanced lead: **{best_f1_model}** has the highest F1 ({comparison.loc[best_f1_model, 'F1']:.3f}) and MCC ({comparison.loc[best_f1_model, 'MCC']:.3f}). "
        f"Highest AUC: **{best_auc_model}** ({comparison.loc[best_auc_model, 'AUC']:.3f})."
    )
else:
    st.success(
        f"Top F1: **{best_f1_model}** ({comparison.loc[best_f1_model, 'F1']:.3f}) · "
        f"Top MCC: **{best_mcc_model}** ({comparison.loc[best_mcc_model, 'MCC']:.3f}) · "
        f"Top AUC: **{best_auc_model}** ({comparison.loc[best_auc_model, 'AUC']:.3f})."
    )

st.divider()
st.markdown('<div class="section-title">Inspect a model</div>', unsafe_allow_html=True)
detail_model = st.selectbox("Choose a model for diagnostics", model_names, index=model_names.index(best_f1_model))
chosen_result = model_results[detail_model]
y_pred = chosen_result["predictions"]
true_positives = sum(actual == 1 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
false_positives = sum(actual == 0 and predicted == 1 for actual, predicted in zip(y_true, y_pred))
false_negatives = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, y_pred))

left_diag, right_diag = st.columns([0.9, 1.1], gap="large")
with left_diag:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_totals = matrix.sum(axis=1)
    heatmap_rows = []
    for row_index, row_label, cell_labels, cell_classes in [
        (0, "Stayed", ["Correct stay", "False alarm"], ["tn", "fp"]),
        (1, "Churned", ["Missed churn", "Correct alert"], ["fn", "tp"]),
    ]:
        cells = []
        for column_index, (cell_label, cell_class) in enumerate(zip(cell_labels, cell_classes)):
            share = matrix[row_index, column_index] / row_totals[row_index] if row_totals[row_index] else 0
            cells.append(
                f'<td class="{cell_class}" title="{cell_label}">'
                f'<span class="heat-value">{matrix[row_index, column_index]:,}</span>'
                f'<span class="heat-share">{share:.0%} of actual {row_label.lower()}</span></td>'
            )
        heatmap_rows.append(f'<tr><th class="row-head">Actual {row_label.lower()}</th>{"".join(cells)}</tr>')
    heatmap_html = (
        '<div class="diagnostic-card"><div class="diagnostic-kicker">Actual outcomes × model predictions</div>'
        '<table class="heatmap"><thead><tr><th></th><th>Predicted · stayed</th>'
        '<th>Predicted · churned</th></tr></thead><tbody>'
        + "".join(heatmap_rows)
        + '</tbody></table><div class="heat-key">'
        '<span class="key-item"><i class="key-swatch good"></i>Correct decisions</span>'
        '<span class="key-item"><i class="key-swatch warn"></i>False alarms</span>'
        '<span class="key-item"><i class="key-swatch miss"></i>Missed churn</span>'
        '</div></div>'
    )
    st.markdown(heatmap_html, unsafe_allow_html=True)
    st.caption(f"Correct churn alerts: {true_positives:,} · False alarms: {false_positives:,} · Missed churners: {false_negatives:,}")
with right_diag:
    st.markdown("#### Classification report")
    report_data = chosen_result["report"]
    report_metrics = ["precision", "recall", "f1-score"]
    report_rows = []
    for class_name, label in [("No churn", "Stayed"), ("Churn", "Churned"), ("macro avg", "Macro avg"), ("weighted avg", "Weighted avg")]:
        row = report_data[class_name]
        score_cells = []
        for metric in report_metrics:
            value = float(row[metric])
            score_cells.append(
                f'<td class="score-cell"><div class="score-value"><span>{value:.2f}</span></div>'
                f'<div class="score-track"><span class="score-fill" style="width:{value * 100:.1f}%"></span></div></td>'
            )
        support = int(row["support"])
        report_rows.append(
            f'<tr><td class="class-name">{label}</td>{"".join(score_cells)}'
            f'<td class="report-support">{support:,}</td></tr>'
        )
    accuracy_value = float(report_data["accuracy"])
    report_html = (
        '<div class="diagnostic-card"><div class="diagnostic-kicker">Per-class quality · positive class is churn</div>'
        '<table class="report-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th>'
        '<th>F1 score</th><th>Support</th></tr></thead><tbody>'
        + "".join(report_rows)
        + f'</tbody></table><div style="display:flex;align-items:center;gap:.65rem;margin:.85rem .15rem .05rem;">'
        f'<span style="background:#d6e5db;border-radius:10px;padding:.55rem .7rem;font-size:.73rem;font-weight:800;color:#28483a;">Overall accuracy</span>'
        f'<span style="background:#fffefa;border:1px solid #d5e0e2;border-radius:10px;padding:.55rem .7rem;font:600 .78rem \'DM Mono\',monospace;color:#28483a;">{accuracy_value:.3f}</span>'
        '</div></div>'
    )
    st.markdown(report_html, unsafe_allow_html=True)

st.markdown("#### Performance notes")
notes = []
for model_name, row in comparison.iterrows():
    leaders = [metric for metric in summary_metrics if comparison[metric].idxmax() == model_name]
    lead_note = f"Leads in {', '.join(leaders)}. " if leaders else ""
    notes.append(
        {
            "Model": model_name,
            "Observation": (
                f"{lead_note}Accuracy {row['Accuracy']:.3f}; AUC {row['AUC']:.3f}; "
                f"precision {row['Precision']:.3f}; recall {row['Recall']:.3f}; "
                f"F1 {row['F1']:.3f}; MCC {row['MCC']:.3f}."
            ),
        }
    )
st.dataframe(pd.DataFrame(notes).set_index("Model"), width="stretch")
st.caption("Models were loaded from saved artifacts. Uploaded rows were used only for evaluation; no training or fitting occurred.")
st.markdown(
    '<div class="footnote">CHURN ATLAS · ML ASSIGNMENT 2 · FIVE SAVED CLASSIFIERS · EVALUATION ONLY</div>',
    unsafe_allow_html=True,
)
