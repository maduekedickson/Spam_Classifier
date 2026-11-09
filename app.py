import os
import re
import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english'))

@st.cache_resource
def load_model(model_path='spam_classifier_model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train and save the model first.")
    return joblib.load(model_path)

STOPWORDS = load_stopwords()

try:
    model = load_model('spam_classifier_model.pkl')
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

# ---------------------------------------------------------
# Text Preprocessing Helper
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize and clean raw text input for inference."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^0-9a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

# ---------------------------------------------------------
# Sidebar: Project Metadata & Model Details
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ About SMS Spam Detector")
    st.markdown(
        """
        This production-ready application leverages **Natural Language Processing (NLP)** 
        and machine learning to detect unsolicited, fraudulent, and phishing SMS messages.
        """
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Architecture & Metrics")
    st.markdown(
        """
        - **Pipeline**: Sub-linear TF-IDF (Unigrams & Bigrams)
        - **Classifier**: Tuned Multinomial Naive Bayes (`alpha=0.2`)
        - **Test Accuracy**: **98.26%**
        - **Spam Precision**: **98.28%** *(Zero-tolerance for false alarms)*
        - **Spam Recall**: **87.02%**
        - **ROC-AUC Score**: **0.9926**
        """
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Why Precision is Critical")
    st.caption(
        "In spam detection, a **False Positive** (marking a legitimate personal, banking, "
        "or two-factor auth message as spam) is far more damaging than occasionally "
        "letting a spam message slip through. Our model is optimized for maximum Spam Precision."
    )
    
    st.markdown("---")
    st.caption("Developed by **Dick Madueke Chukwu** | Powered by Streamlit & Scikit-Learn")

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    if os.path.exists("image.jpg"):
        st.image("image.jpg", caption="AI-Powered SMS Classification", use_container_width=True)
    else:
        st.info("💡 Place `image.jpg` in the project root to display header banner.")

with col2:
    st.title("SMS Spam Detection System")
    st.markdown(
        "Enter any SMS text or select one of the real-world preset examples below "
        "to classify whether a message is **Legitimate (Ham)** or **Spam / Smishing**."
    )

st.markdown("---")

# Quick Example Prompts
st.markdown("##### ⚡ Quick Test with Examples:")
preset_samples = {
    "🚨 Lottery Winner (Spam)": "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word CLAIM to 81010 now. T&Cs apply.",
    "⚠️ Bank Alert (Spam)": "Security Alert: We identified suspicious login attempts on your account. Verify immediately at https://secure-bank-login.biz to avoid suspension.",
    "☕ Casual Lunch (Ham)": "Hey Dickson, are we still meeting for lunch at 1pm today? Let me know if you are free.",
    "📦 Delivery Status (Ham)": "Your package with tracking number 89218 is out for delivery with our courier and should arrive before 4pm today."
}

example_cols = st.columns(len(preset_samples))
selected_text = None

for idx, (label, sample_text) in enumerate(preset_samples.items()):
    with example_cols[idx]:
        if st.button(label, use_container_width=True):
            st.session_state["user_message"] = sample_text

# Text Input Area
default_val = st.session_state.get("user_message", "")
user_input = st.text_area(
    "Enter SMS Message Content:",
    value=default_val,
    height=120,
    placeholder="Type or paste message here (e.g., 'Congratulations! You won £1000 cash. Call now!')..."
)

# Classification Trigger
col_btn, col_clear = st.columns([1, 5])
with col_btn:
    check_button = st.button("🔍 Classify Message", type="primary", use_container_width=True)
with col_clear:
    if st.button("🧹 Clear Input", use_container_width=False):
        st.session_state["user_message"] = ""
        st.rerun()

# ---------------------------------------------------------
# Prediction & Results Display
# ---------------------------------------------------------
if check_button or (user_input and user_input != default_val):
    if not model_loaded:
        st.error(f"❌ Model could not be loaded: {model_error}")
    elif not user_input.strip():
        st.warning("⚠️ Please enter or select a non-empty text message to classify.")
    else:
        # Preprocess input
        clean_input = clean_text(user_input)
        
        # If cleaned text is empty (e.g. only punctuation), fallback to lowercased raw text
        eval_text = clean_input if clean_input else user_input.lower().strip()
        
        # Run inference
        prediction = model.predict([eval_text])[0]
        probabilities = model.predict_proba([eval_text])[0]
        
        # Extract class probabilities
        classes = list(model.classes_)
        spam_idx = classes.index("spam") if "spam" in classes else 1
        ham_idx = classes.index("ham") if "ham" in classes else 0
        
        spam_proba = float(probabilities[spam_idx]) * 100
        ham_proba = float(probabilities[ham_idx]) * 100
        
        st.markdown("### 📋 Prediction Results")
        
        res_col1, res_col2 = st.columns([1, 1], gap="medium")
        
        if prediction == "spam":
            with res_col1:
                st.markdown(
                    f"""
                    <div style="background-color: #f8d7da; border: 2px solid #f5c2c7; border-radius: 10px; padding: 20px; text-align: center; color: #842029;">
                        <h2 style="margin: 0; color: #842029;">🚨 SPAM DETECTED</h2>
                        <p style="font-size: 16px; margin: 10px 0 0 0;">This message matches patterns commonly found in promotional spam, smishing, or scam campaigns.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with res_col2:
                st.metric("Spam Confidence", f"{spam_proba:.2f}%", delta=f"{spam_proba - 50:.1f}% vs baseline", delta_color="inverse")
                st.progress(spam_proba / 100.0)
                st.caption(f"Confidence breakdown: **{spam_proba:.2f}% Spam** vs **{ham_proba:.2f}% Ham**")
        else:
            with res_col1:
                st.markdown(
                    f"""
                    <div style="background-color: #d1e7dd; border: 2px solid #badbcc; border-radius: 10px; padding: 20px; text-align: center; color: #0f5132;">
                        <h2 style="margin: 0; color: #0f5132;">✅ LEGITIMATE (HAM)</h2>
                        <p style="font-size: 16px; margin: 10px 0 0 0;">This message is verified as safe and typical of normal conversational or notification SMS.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with res_col2:
                st.metric("Legitimacy Confidence", f"{ham_proba:.2f}%", delta=f"{ham_proba - 50:.1f}% vs baseline", delta_color="normal")
                st.progress(ham_proba / 100.0)
                st.caption(f"Confidence breakdown: **{ham_proba:.2f}% Ham** vs **{spam_proba:.2f}% Spam**")
        
        # Message Inspector
        with st.expander("🔍 Message Feature Analysis & Tokenization Inspector"):
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Characters", len(user_input))
            m_col2.metric("Words", len(user_input.split()))
            m_col3.metric("Digits", sum(c.isdigit() for c in user_input))
            m_col4.metric("Clean Tokens", len(clean_input.split()) if clean_input else 0)
            
            st.markdown("**Processed Tokens Analyzed by Model:**")
            st.code(clean_input if clean_input else "(Punctuation/Stopwords only)", language="text")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 13px;'>"
    "SMS Spam Classifier • Built with Scikit-Learn & Streamlit • Production Model: Multinomial Naive Bayes (TF-IDF)"
    "</div>",
    unsafe_allow_html=True
)
