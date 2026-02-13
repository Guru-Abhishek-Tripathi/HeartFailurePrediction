import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="🫀",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS — Dark Navy Theme
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #070d1a;
    color: #c8d6f0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Header Banner ── */
.banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a2a5e 50%, #0d1f3c 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,120,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.banner-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
}
.banner-sub {
    font-size: 0.95rem;
    color: #7a9fd4;
    margin: 0;
    line-height: 1.6;
}
.banner-badge {
    display: inline-block;
    background: rgba(0,120,255,0.15);
    border: 1px solid rgba(0,120,255,0.3);
    color: #4d9fff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 1px;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4d9fff;
    border-left: 3px solid #4d9fff;
    padding-left: 10px;
    margin: 1.6rem 0 0.8rem 0;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: #0d1f3c;
    border: 1px solid #1e3a6e;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4d9fff;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1e3a6e;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }

.sidebar-logo {
    text-align: center;
    padding: 1rem 0 1.4rem 0;
    border-bottom: 1px solid #1e3a6e;
    margin-bottom: 1.2rem;
}
.sidebar-logo-icon { font-size: 2.4rem; line-height: 1; }
.sidebar-logo-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #4d9fff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3d5a8a;
    margin: 1.2rem 0 0.4rem 0;
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    background: #0d1f3c !important;
    border: 1px dashed #1e3a6e !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div {
    background: #0d1f3c !important;
    border: 1px solid #1e3a6e !important;
    border-radius: 8px !important;
    color: #c8d6f0 !important;
}
.stButton > button {
    background: #0a2a5e !important;
    border: 1px solid #1e5aaa !important;
    color: #c8d6f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #1e3a6e !important;
    border-color: #4d9fff !important;
    color: #ffffff !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(0,90,200,0.15) !important;
    border: 1px solid #1e5aaa !important;
    color: #4d9fff !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3a6e !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Info / Error boxes ── */
[data-testid="stAlert"] {
    background: #0d1f3c !important;
    border-color: #1e3a6e !important;
    border-radius: 10px !important;
    color: #c8d6f0 !important;
}

/* ── Footer ── */
.footer {
    margin-top: 3rem;
    padding: 1.2rem 2rem;
    border-top: 1px solid #1e3a6e;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: #3d5a8a;
    font-family: 'Space Mono', monospace;
}
.footer-name { color: #4d9fff; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Model & Scaler Loading
# -------------------------------------------------
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    models = {}
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl") and file != "scaler.pkl":
            name = file.replace(".pkl", "").replace("_", " ")
            models[name] = joblib.load(os.path.join(MODEL_DIR, file))
    return models

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

models = load_models()
scaler = load_scaler()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🫀</div>
        <div class="sidebar-logo-title">HF Predict</div>
    </div>
    """, unsafe_allow_html=True)

    # Step 1 — Download sample data
    st.markdown('<div class="sidebar-section">Step 1 — Sample Data</div>', unsafe_allow_html=True)
    TEST_DATA_PATH = os.path.join("models","data", "test_data.csv")
    if os.path.exists(TEST_DATA_PATH):
        with open(TEST_DATA_PATH, "rb") as f:
            st.download_button(
                label="⬇ Download Test CSV",
                data=f,
                file_name="test_data.csv",
                mime="text/csv"
            )
    else:
        st.caption("test_data.csv not found in /data")

    # Step 2 — Upload
    st.markdown('<div class="sidebar-section">Step 2 — Upload Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    # Step 3 — Select Model
    st.markdown('<div class="sidebar-section">Step 3 — Select Model</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Model", list(models.keys()),
        disabled=(uploaded_file is None),
        label_visibility="collapsed"
    )

    # Step 4 — Actions
    st.markdown('<div class="sidebar-section">Step 4 — Actions</div>', unsafe_allow_html=True)
    run_button   = st.button("▶  RUN ANALYSIS", disabled=(uploaded_file is None))
    clear_button = st.button("✕  CLEAR",        disabled=(uploaded_file is None))

    if clear_button:
        st.session_state.clear()
        st.rerun()

# -------------------------------------------------
# Header Banner
# -------------------------------------------------
st.markdown("""
<div class="banner">
    <div class="banner-badge">BITS PILANI · M.TECH AIML · ML ASSIGNMENT 2</div>
    <div class="banner-title">🫀 Heart Failure Prediction</div>
    <p class="banner-sub">
        Upload a test dataset and evaluate six ML classification models —
        Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest &amp; XGBoost.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Main Content
# -------------------------------------------------
if uploaded_file is None:
    st.markdown("""
    <div style="
        background:#0d1f3c;
        border:1px dashed #1e3a6e;
        border-radius:14px;
        padding:2.5rem;
        text-align:center;
        color:#3d5a8a;
        font-family:'Space Mono',monospace;
        font-size:0.85rem;
        letter-spacing:1px;
    ">
        &#8592; DOWNLOAD SAMPLE DATA &nbsp;&middot;&nbsp; UPLOAD CSV &nbsp;&middot;&nbsp;
        SELECT MODEL &nbsp;&middot;&nbsp; RUN
    </div>
    """, unsafe_allow_html=True)

else:
    df = pd.read_csv(uploaded_file)

    st.markdown('<div class="section-heading">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"{df.shape[0]} rows · {df.shape[1]} columns")

    if df.shape[1] < 2:
        st.error("Dataset must contain at least one feature column and a target column.")
        st.stop()

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1]

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().sum().sum() > 0:
        st.error("Dataset contains invalid values after encoding. Please check your CSV.")
        st.stop()

    try:
        X_scaled = scaler.transform(X)
    except Exception:
        st.error("Feature mismatch: uploaded CSV columns don't match the trained model's expected features.")
        st.stop()

    selected_model = models[selected_model_name]

    if run_button:
        y_pred = selected_model.predict(X_scaled)

        auc = None
        if hasattr(selected_model, "predict_proba"):
            y_proba = selected_model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_proba)

        accuracy  = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall    = recall_score(y, y_pred, zero_division=0)
        f1        = f1_score(y, y_pred, zero_division=0)
        mcc       = matthews_corrcoef(y, y_pred)

        # ── Metrics ──
        st.markdown('<div class="section-heading">Evaluation Metrics</div>', unsafe_allow_html=True)
        auc_display = f"{auc:.4f}" if auc is not None else "N/A"

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{accuracy:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{precision:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{recall:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{f1:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AUC Score</div>
                <div class="metric-value">{auc_display}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MCC</div>
                <div class="metric-value">{mcc:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confusion Matrix + Per-Class ──
        col_left, col_right = st.columns([1, 1.6])

        with col_left:
            st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor("#0d1f3c")
            ax.set_facecolor("#0d1f3c")
            ax.imshow(cm, cmap="Blues", vmin=0)

            labels = [
                [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
                [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]
            ]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, labels[i][j],
                            ha="center", va="center",
                            fontsize=11, fontweight="bold", color="white")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["No Disease", "Disease"], color="#7a9fd4", fontsize=9)
            ax.set_yticklabels(["No Disease", "Disease"], color="#7a9fd4", fontsize=9)
            ax.set_xlabel("Predicted", color="#4d9fff", fontsize=9)
            ax.set_ylabel("Actual",    color="#4d9fff", fontsize=9)
            ax.tick_params(colors="#7a9fd4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e3a6e")
            plt.tight_layout()
            st.pyplot(fig)

        with col_right:
            st.markdown('<div class="section-heading">Per-Class Performance</div>', unsafe_allow_html=True)

            report  = classification_report(y, y_pred, output_dict=True)
            class_0 = report['0']
            class_1 = report['1']

            class_report = pd.DataFrame({
                'Precision': [class_0['precision'], class_1['precision']],
                'Recall':    [class_0['recall'],    class_1['recall']],
                'F1-Score':  [class_0['f1-score'],  class_1['f1-score']],
                'Support':   [int(class_0['support']), int(class_1['support'])]
            }, index=['Class 0 — No Disease', 'Class 1 — Disease']).round(4)

            st.dataframe(class_report, use_container_width=True)

            st.markdown('<div class="section-heading">Interpretation</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Precision**\n\n{class_1['precision']*100:.1f}% of predicted disease cases were correct.")
            c2.info(f"**Recall**\n\n{class_1['recall']*100:.1f}% of actual disease cases were caught.")
            c3.info(f"**F1-Score**\n\n{class_1['f1-score']:.4f} — harmonic mean of precision & recall.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("""
<div class="footer">
    <span>BITS PILANI &nbsp;·&nbsp; WILP &nbsp;·&nbsp; M.Tech AIML &nbsp;·&nbsp; ML Assignment 2</span>
    <span class="footer-name">Abhishek Tiwari - 2025AA05131</span>
</div>
""", unsafe_allow_html=True)