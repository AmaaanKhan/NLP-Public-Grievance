
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Public Grievance Analytics",
    page_icon="🏙️",
    layout="wide"
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "grievance_classifier.joblib"
OUTPUT_DIR = ROOT / "outputs"

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 16px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_outputs():
    distribution = pd.read_csv(OUTPUT_DIR / "category_distribution.csv")
    with open(OUTPUT_DIR / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)
    report = pd.read_json(OUTPUT_DIR / "classification_report.json").T
    return distribution, metrics, report


st.sidebar.title("Public Grievance NLP")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Grievance Classifier", "Model Evaluation"]
)

st.sidebar.divider()
st.sidebar.caption("Dataset: BBMP Bengaluru citizen grievances")
st.sidebar.caption("Model: TF-IDF + Logistic Regression")

distribution, metrics, report = load_outputs()

if page == "Overview":
    st.title("Public Grievance Analysis")
    st.write(
        "An NLP-based prototype for analysing civic grievance categories "
        "and predicting the department associated with a grievance subcategory."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grievance records", f"{sum(distribution['Count']):,}")
    c2.metric("Categories", metrics["categories"])
    c3.metric("Model accuracy", f"{metrics['accuracy']:.2%}")
    c4.metric("Macro F1-score", f"{metrics['macro_f1']:.2%}")

    st.subheader("Grievances by Category")
    top_n = st.slider("Categories to display", 5, len(distribution), 15)
    top_categories = distribution.head(top_n).sort_values("Count")
    fig = px.bar(
        top_categories,
        x="Count",
        y="Category",
        orientation="h",
        color="Count",
        color_continuous_scale="Teal",
        labels={"Count": "Number of records", "Category": "Grievance category"},
        template="plotly_white"
    )
    fig.update_layout(coloraxis_showscale=False, height=max(420, top_n * 28))
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The source contains structured grievance categories and subcategories, "
        "not full citizen-written complaint narratives. The classifier predicts "
        "Category from Sub Category."
    )

elif page == "Grievance Classifier":
    st.title("Grievance Category Classifier")
    st.write("Enter a grievance subcategory or short civic issue description.")

    examples = [
        "Street Light Not Working",
        "Garbage vehicle not arrived",
        "Potholes",
        "Stray dog related complaints",
        "Removal of dead/fallen trees",
        "Dengue positive"
    ]

    selected = st.selectbox("Try an example", ["Custom input"] + examples)
    default_text = "" if selected == "Custom input" else selected
    text = st.text_input("Grievance text", value=default_text)

    if st.button("Predict category", type="primary"):
        if not text.strip():
            st.warning("Enter a grievance subcategory first.")
        else:
            model = load_model()
            prediction = model.predict([text.strip()])[0]
            probabilities = model.predict_proba([text.strip()])[0]
            classes = model.named_steps["classifier"].classes_
            ranking = sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True
            )[:5]

            st.success(f"Predicted category: **{prediction}**")
            st.metric("Model confidence", f"{max(probabilities):.1%}")
            st.caption(
                "Confidence is the model's estimated probability, not a guarantee "
                "that the prediction is correct."
            )

            st.subheader("Top predicted categories")
            results = pd.DataFrame(ranking, columns=["Category", "Probability"])
            results["Probability"] *= 100
            fig = px.bar(
                results.sort_values("Probability"),
                x="Probability",
                y="Category",
                orientation="h",
                labels={"Probability": "Estimated probability (%)"},
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write("Evaluation on the held-out test set.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
    c3.metric("Weighted F1", f"{metrics['weighted_f1']:.2%}")

    st.subheader("Per-category performance")
    display_cols = [
        column for column in ["precision", "recall", "f1-score", "support"]
        if column in report.columns
    ]
    category_report = report.loc[
        ~report.index.isin(["accuracy", "macro avg", "weighted avg"]),
        display_cols
    ]
    st.dataframe(category_report, use_container_width=True)

    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    if matrix_path.exists():
        st.subheader("Normalized Confusion Matrix")
        st.image(str(matrix_path), use_container_width=True)

    st.caption(
        "The model uses TF-IDF features extracted from Sub Category text. "
        "Results measure prediction of the existing Category label, not "
        "classification of full citizen-written complaint narratives."
    )

st.divider()
st.caption("Academic prototype • BBMP citizen grievance dataset")
