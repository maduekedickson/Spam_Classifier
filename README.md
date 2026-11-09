# 🛡️ SMS Spam Detection System (NLP)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20NLTK-green)](https://www.nltk.org/)

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to identify fraudulent, phishing (**smishing**), and promotional SMS spam messages with high precision and low latency. The application is packaged in a self-contained Scikit-Learn `Pipeline` and deployed as an interactive web interface powered by Streamlit Cloud.

---

## 📌 Problem Statement & Business Context

Short Message Service (SMS) is one of the highest-engagement communication channels globally, with open rates exceeding 95%. However, mobile users are continually targeted by financial scams, lottery fraud, phishing URLs, and aggressive advertising.

When engineering a spam classification system for production, two challenges dominate:
1. **Asymmetric Error Costs**: A **False Positive** (classifying an important banking alert, OTP code, or personal message as spam) damages user trust far more than a **False Negative** (allowing an occasional spam message into the inbox). Consequently, **Spam Precision** is the primary metric to optimize alongside F1-Score.
2. **Severe Class Imbalance**: Natural SMS traffic typically exhibits an 85–90% legitimate (ham) to 10–15% spam ratio. Naive accuracy alone is misleading; stratified validation and threshold analysis are essential.

---

## ✨ Key Features

- **High-Precision NLP Pipeline**: Uses sublinear TF-IDF vectorization with unigram and bigram extraction, followed by a tuned Multinomial Naive Bayes classifier (`alpha=0.2`).
- **Real-Time Probability Breakdown**: Computes confidence scores via `predict_proba()` to present visual certainty meters (e.g., *99.7% Spam* vs *0.3% Ham*).
- **One-Click Test Presets**: Built-in sample buttons (Lottery scams, Bank security alerts, Casual lunch messages, Courier tracking) for instant demonstration.
- **Message Token & Metric Inspector**: Expandable inspection panel detailing character count, word count, digit frequency, and normalized model tokens.
- **Optimized Caching & Performance**: Employs `@st.cache_resource` to load the serialized model once in memory and `@st.cache_data` for NLTK stopword handling, ensuring sub-100ms response times.
- **Self-Contained Deployment**: The saved pipeline encapsulates text vectorization and classification in a single object (`spam_classifier_model.pkl`), preventing training-serving skew.

---

## 📊 Dataset Overview

The model was developed on the benchmark **SMS Spam Collection Dataset** (`spam.csv`):

| Metric | Raw Dataset | Cleaned & Deduplicated |
| :--- | :--- | :--- |
| **Total Messages** | 5,572 | **5,169** |
| **Legitimate (Ham)** | 4,825 (86.6%) | **4,516 (87.37%)** |
| **Spam Messages** | 747 (13.4%) | **653 (12.63%)** |
| **Duplicate Entries** | 403 (7.23%) | **0 (Purged)** |

### Key Exploratory Insights
- **Character Length**: Ham messages average **70.5 characters** (median: 52), whereas Spam messages cluster tightly at **137.9 characters** (median: 149), maximizing SMS carrier limits (160 chars).
- **Digit Density**: Spam messages contain an average of **15.6 numeric digits** (phone numbers, monetary prizes, coupon codes) compared to **0.3 digits** in Ham messages.
- **Lexical Signals**: Spam messages are dominated by urgency and transactional terms (`call`, `free`, `txt`, `claim`, `prize`, `won`, `urgent`), while Ham messages consist of relational vocabulary (`u`, `ok`, `go`, `come`, `like`, `know`).

---

## 🏆 Model Benchmark & Evaluation

Four candidate architectures were trained and evaluated using **5-Fold Stratified Cross-Validation** on the training set (80%) and scored on the holdout test set (20% = 1,034 samples):

| Model Architecture | 5-Fold CV Macro F1 | Test Accuracy | Test Precision (Spam) | Test Recall (Spam) | Test F1 (Spam) | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Champion)** | **0.9265** | **98.26%** | **98.28%** | **87.02%** | **0.9231** | **0.9926** |
| **Logistic Regression (L2)** | 0.8101 | 96.03% | 98.91% | 69.47% | 0.8161 | 0.9918 |
| **Calibrated Linear SVC** | 0.9292 | 97.58% | 92.74% | 87.79% | 0.9020 | 0.9933 |
| **Random Forest (100 Trees)** | 0.8865 | 97.49% | 98.17% | 81.68% | 0.8917 | 0.9889 |

### Confusion Matrix on Unseen Test Set (1,034 Messages)
```
                  Predicted Ham      Predicted Spam
Actual Ham            901                  2        (99.78% True Negative Rate)
Actual Spam            17                114        (87.02% Detection Rate)
```
- **False Positives**: Only **2 out of 903** legitimate messages were flagged, satisfying the strict low-false-alarm requirement.
- **Inference Speed**: Microsecond inference time with a lightweight serialized model footprint (~340 KB).

---

## 📁 Repository Structure

```text
Spam_Classifier/
├── app.py                     # Production Streamlit application
├── spam_classifier_model.pkl  # Serialized Scikit-Learn TF-IDF + MNB pipeline
├── Spam Detection.ipynb       # End-to-end reproducible research & analysis notebook
├── spam.csv                   # SMS Spam Collection dataset
├── requirements.txt           # Production environment dependencies
├── .gitignore                 # Standard Python & Jupyter exclusion rules
├── image.jpg                  # Header banner asset for web application
└── README.md                  # Comprehensive project documentation
```

---

## 🚀 Local Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/maduekedickson/Spam_Classifier.git
cd Spam_Classifier
```

### 2. Create and Activate a Virtual Environment
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Cloud)

1. Fork or push this repository to your GitHub account: `https://github.com/<your-username>/Spam_Classifier`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**.
4. Select your repository, set the branch to `main`, and specify the main file path as `app.py`.
5. Click **"Deploy"**. Streamlit Cloud will automatically install dependencies from `requirements.txt` and launch the app.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn (Pipelines, MultinomialNB, TF-IDF, Metrics)
- **Natural Language Processing**: NLTK (Stopwords, Text Normalization)
- **Data Manipulation**: Pandas, NumPy
- **Visual Analytics**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: Joblib
