import streamlit as st
import numpy as np
import joblib
import PyPDF2
import shap
import plotly.graph_objects as go

from modules.pre_processing import preprocess_essay
from modules.model_load import model, FEATURE_ORDER


# ==============================
# PAGE CONFIG (Default Light Theme)
# ==============================
st.set_page_config(
    page_title="AI Essay Detector",
    page_icon="📝",
    layout="wide",
)


# ==============================
# HUMAN-FRIENDLY EXPLANATIONS
# ==============================
FRIENDLY_RULES = {
    "punct_ratio": {
        "high": "Your essay uses more punctuation than typical human writing.",
        "low": "Your essay uses less punctuation, making it feel more mechanical."
    },
    "avg_token_length": {
        "high": "Your essay uses longer words, which is common in AI-generated text.",
        "low": "Your essay uses shorter, simpler words typical of human writing."
    },
    "type_token_ratio": {
        "high": "Your vocabulary is unusually diverse — AI models often force variety.",
        "low": "You repeat words more naturally, like human writers do."
    },
    "hapax_legomena_ratio": {
        "high": "Your essay contains many one-time words — AI often does this.",
        "low": "Consistent vocabulary suggests natural human writing."
    },
    "pos_adv_ratio": {
        "high": "More adverbs — common in human expressive writing.",
        "low": "Few adverbs — more robotic or AI-like."
    },
    "pos_adj_ratio": {
        "high": "Your writing is descriptive (human trait).",
        "low": "Low adjective usage — AI sometimes under-describes."
    },
    "pos_verb_ratio": {
        "high": "High verb usage — strong sign of expressive human writing.",
        "low": "Low verb usage — robotic or info-style AI tone."
    },
    "pos_pron_ratio": {
        "high": "Many pronouns — typical in human writing.",
        "low": "Few pronouns — AI often avoids personal tone."
    },
    "sentence_token_entropy": {
        "high": "Vocabulary randomness is high — AI often forces variety.",
        "low": "Low randomness — humans write more consistently."
    },
}

def explain_feature(fname, shap_value):
    direction = "high" if shap_value > 0 else "low"
    return FRIENDLY_RULES.get(fname, {}).get(direction, "This feature influenced the decision.")

def shap_color(value):
    return "🟥 AI-like" if value > 0 else "🟩 Human-like"


# ==============================
# HELPERS
# ==============================
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages])

def shap_explain(vector):
    xgb = model.named_steps["xgb"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(vector)
    return shap_values[0]

def top_contributors(shap_values, top_n=3):
    return np.argsort(np.abs(shap_values))[::-1][:top_n]


# ==============================
# HEADER
# ==============================
st.title("📝 AI Essay Detector")
st.write("Upload a file or paste an essay to check if it's **AI-generated** or **Human-written**.")


# ==============================
# INPUT AREA
# ==============================
left, right = st.columns(2)

uploaded_file = left.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
text_input = right.text_area("Or paste your essay here:", height=250)

text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")
elif text_input.strip():
    text = text_input.strip()


# ==============================
# ANALYSIS
# ==============================
if st.button("🔍 Analyze Essay"):

    if text.strip() == "":
        st.error("Please upload or paste text.")
    else:
        features = preprocess_essay(text)
        vector = np.array([features[f] for f in FEATURE_ORDER]).reshape(1, -1)

        pred = model.predict(vector)[0]
        prob_ai = model.predict_proba(vector)[0][1]
        prob_human = 1 - prob_ai

        label = "AI Generated" if pred == 1 else "Human Written"

        # METRICS
        c1, c2, c3 = st.columns(3)
        c1.metric("Classification", label)
        c2.metric("AI Probability", f"{prob_ai * 100:.2f}%")
        c3.metric("Human Confidence", f"{prob_human * 100:.2f}%")

        st.markdown("---")

        # CONFIDENCE GAUGE
        st.subheader("📏 Confidence Gauge")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_ai * 100,
            title={'text': "AI Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4d4d"},
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},
                    {'range': [40, 70], 'color': "#f1c40f"},
                    {'range': [70, 100], 'color': "#e74c3c"},
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("---")

        # TOP 3 REASONS
        st.subheader("🧠 Why This Classification? (Top 3 Reasons)")

        shap_vals = shap_explain(vector)
        top_idx = top_contributors(shap_vals)

        for idx in top_idx:
            fname = FEATURE_ORDER[idx]
            val = shap_vals[idx]
            st.markdown(f"""
            ### 🔹 {fname.replace('_',' ').title()}
            **Impact:** `{val:.4f}` {shap_color(val)}  
            **Explanation:** {explain_feature(fname, val)}
            """)

        # SUMMARY
        st.markdown("### 📘 Summary (Simple Explanation)")

        if pred == 1:
            st.error(f"""
            Your essay shows strong **AI-like writing patterns**, including:
            • {explain_feature(FEATURE_ORDER[top_idx[0]], shap_vals[top_idx[0]])}  
            • {explain_feature(FEATURE_ORDER[top_idx[1]], shap_vals[top_idx[1]])}  
            • {explain_feature(FEATURE_ORDER[top_idx[2]], shap_vals[top_idx[2]])}

            These indicate **AI-generated text**.
            """)
        else:
            st.success(f"""
            Your essay displays strong **human writing characteristics**, such as:
            • {explain_feature(FEATURE_ORDER[top_idx[0]], shap_vals[top_idx[0]])}  
            • {explain_feature(FEATURE_ORDER[top_idx[1]], shap_vals[top_idx[1]])}  
            • {explain_feature(FEATURE_ORDER[top_idx[2]], shap_vals[top_idx[2]])}

            This strongly suggests **human-written content**.
            """)
