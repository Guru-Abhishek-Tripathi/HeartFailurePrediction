import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
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

st.title("🫀 Heart Failure Prediction App")
st.markdown("Upload a **test dataset (CSV)** and evaluate trained machine learning models.")

MODEL_DIR = "models"

# -------------------------------------------------
# Load Models and Scaler
# -------------------------------------------------
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
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("⚙ Controls")

uploaded_file = st.sidebar.file_uploader(
    "1️⃣ Upload Test Dataset (CSV)",
    type=["csv"]
)
# Download sample test data button 
TEST_DATA_PATH = os.path.join("models", "data", "test_data.csv")
if os.path.exists(TEST_DATA_PATH):
    with open(TEST_DATA_PATH, "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Download Sample Test Data",
            data=f,
            file_name="test_data.csv",
            mime="text/csv",
            help="Download the held-out test split to try the app"
        )


selected_model_name = st.sidebar.selectbox(
    "2️⃣ Select Model",
    list(models.keys()),
    disabled=(uploaded_file is None)
)

# Better UI: Run + Clear Buttons Side-by-Side
colA, colB = st.sidebar.columns(2)
run_button = run_button = colA.button("▶ Run", disabled=(uploaded_file is None))

clear_button = colB.button("🔄 Clear",disabled=(uploaded_file is None))

# Clear Button Logic
if clear_button:
    st.session_state.clear()
    st.experimental_rerun()

# -------------------------------------------------
# Main App Logic
# -------------------------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    if df.shape[1] < 2:
        st.error("Dataset must contain features and target column.")
        st.stop()

    # Separate features and target (last column assumed target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # -------------------------------------------------
    # Handle Categorical Columns
    # -------------------------------------------------
    cat_cols = X.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().sum().sum() > 0:
        st.error("Dataset contains invalid values after encoding.")
        st.stop()

    # -------------------------------------------------
    # Apply StandardScaler (same as training)
    # -------------------------------------------------
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        st.error("Feature mismatch with trained model.")
        st.stop()

    selected_model = models[selected_model_name]

    # -------------------------------------------------
    # Run Evaluation
    # -------------------------------------------------
    if run_button:

        y_pred = selected_model.predict(X_scaled)

        if hasattr(selected_model, "predict_proba"):
            y_proba = selected_model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, y_proba)
        else:
            auc = None

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y, y_pred)

        # -------------------------------------------------
        # Metrics Display
        # -------------------------------------------------
        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Precision", f"{precision:.4f}")

        col2.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")

        col3.metric("MCC", f"{mcc:.4f}")
        if auc is not None:
            col3.metric("AUC", f"{auc:.4f}")

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------
        st.subheader("🧮 Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j],
                        ha="center", va="center",
                        fontsize=14, fontweight="bold")

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])

        st.pyplot(fig)

        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------
        st.subheader("📄 Per-Class Performance")

        report = classification_report(y, y_pred, output_dict=True)
        
        # Extract only the per-class metrics (exclude accuracy, macro avg, weighted avg)
        class_0 = report['0']
        class_1 = report['1']
        
        # Create a clean dataframe with only per-class metrics
        class_report = pd.DataFrame({
            'Precision': [class_0['precision'], class_1['precision']],
            'Recall': [class_0['recall'], class_1['recall']],
            'F1-Score': [class_0['f1-score'], class_1['f1-score']],
            'Support': [int(class_0['support']), int(class_1['support'])]
        }, index=['Class 0 (No Disease)', 'Class 1 (Disease)'])
        
        # Round to 4 decimal places
        class_report = class_report.round(4)
        
        st.dataframe(class_report, use_container_width=True)

        # Add detailed interpretation
        st.markdown("---")
        st.subheader("📊 Metrics Explained")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Precision (Disease):** {class_1['precision']:.4f}
            
            Of all predictions saying "Disease", 
            {class_1['precision']*100:.2f}% were correct.
            """)
        
        with col2:
            st.info(f"""
            **Recall (Disease):** {class_1['recall']:.4f}
            
            Of all actual disease cases, 
            {class_1['recall']*100:.2f}% were identified.
            """)
        
        with col3:
            st.info(f"""
            **F1-Score (Disease):** {class_1['f1-score']:.4f}
            
            Balanced score between 
            precision and recall.
            """)

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
