# 📝 AI Essay Detector

An intelligent machine-learning web app that detects whether an essay is **AI-generated** or **Human-written**, using linguistic features, POS ratios, entropy, and SHAP-based explainability.

Built with **Python, Streamlit, XGBoost, spaCy**, and **SHAP**.

Try it out here -> http://humanorai.streamlit.app/
---



## 🚀 Features

### 🔍 AI vs Human Essay Classification
Detect whether an essay is written by a human or an AI model using an XGBoost classifier trained on linguistic features.

### 📊 Explainable AI (XAI)
Provides:
- Top 3 most influential features  
- Human-friendly explanations  
- Confidence percentages  
- Clear SHAP-based reasoning  

### 📄 Input Support
- Upload **PDF** or **TXT**
- Paste text manually

### 🧠 Linguistic Feature Engineering
Extracts metrics like:
- Punctuation ratio  
- Average word length  
- POS tag ratios  
- Vocabulary richness  
- Token entropy  
- Sentence entropy  

### 📏 Visual Confidence Gauge
Interactive gauge meter shows AI probability intuitively.

--
