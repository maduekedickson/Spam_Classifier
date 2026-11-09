"""
train.py - Production Model Training Script for SMS Spam Classifier
===================================================================
This script trains and serializes the champion NLP pipeline (TF-IDF +
Multinomial Naive Bayes) for SMS spam detection.

Usage:
    python train.py
    python train.py --data_path spam.csv --model_path spam_classifier_model.pkl --alpha 0.2
"""

import argparse
import os
import re
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def normalize_text(text: str) -> str:
    """Normalize raw text by lowercasing, stripping special characters, and collapsing whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """Load raw dataset, clean columns, and purge duplicate entries."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")

    raw_df = pd.read_csv(data_path, encoding='latin1')
    
    # Retain and semantically rename primary columns
    df = raw_df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'}).copy()
    
    # Audit duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['message'], keep='first').reset_index(drop=True)
    deduped_count = len(df)
    print(f"[Data] Loaded {initial_count} rows -> Purged {initial_count - deduped_count} duplicates -> {deduped_count} clean records.")
    
    # Preprocess text
    df['clean_text'] = df['message'].apply(normalize_text)
    
    # Print class balance
    class_dist = df['label'].value_counts(normalize=True) * 100
    print(f"[Data] Class Balance: Ham={class_dist.get('ham', 0):.1f}%, Spam={class_dist.get('spam', 0):.1f}%")
    
    return df


def build_pipeline(alpha: float = 0.2, max_features: int = 5000) -> Pipeline:
    """Construct Scikit-Learn TF-IDF + Multinomial Naive Bayes pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', MultinomialNB(alpha=alpha))
    ])


def evaluate_pipeline(pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series):
    """Perform 5-fold cross-validation and evaluate on holdout test set."""
    print("\n[Cross-Validation] Running 5-Fold Stratified CV on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"[Cross-Validation] Mean Macro F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Fit on training split
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec_spam = precision_score(y_test, y_pred, pos_label='spam')
    rec_spam = recall_score(y_test, y_pred, pos_label='spam')
    f1_spam = f1_score(y_test, y_pred, pos_label='spam')

    print("\n" + "=" * 50)
    print("Holdout Test Set Performance (N = {})".format(len(y_test)))
    print("=" * 50)
    print(f"Accuracy:              {acc * 100:.2f}%")
    print(f"Precision (Spam):      {prec_spam * 100:.2f}% (False Positive Resistance)")
    print(f"Recall (Spam):         {rec_spam * 100:.2f}%")
    print(f"F1-Score (Spam):       {f1_spam:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=['ham', 'spam']))
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))


def main():
    parser = argparse.ArgumentParser(description="Train and serialize SMS Spam Classifier NLP pipeline.")
    parser.add_argument("--data_path", type=str, default="spam.csv", help="Path to input spam.csv dataset.")
    parser.add_argument("--model_path", type=str, default="spam_classifier_model.pkl", help="Output path for serialized model.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Laplace smoothing parameter for MultinomialNB.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Holdout test split ratio (default: 0.2).")
    args = parser.parse_args()

    print("=" * 60)
    print("SMS Spam Classifier - Production Training Workflow")
    print("=" * 60)

    # 1. Load data
    df = load_and_preprocess_data(args.data_path)

    # 2. Stratified train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=args.test_size,
        random_state=42,
        stratify=df['label']
    )

    # 3. Build & evaluate candidate pipeline
    eval_pipeline = build_pipeline(alpha=args.alpha)
    evaluate_pipeline(eval_pipeline, X_train, y_train, X_test, y_test)

    # 4. Retrain production model on 100% of deduplicated data
    print("\n[Production] Retraining final pipeline on all", len(df), "deduplicated messages...")
    prod_pipeline = build_pipeline(alpha=args.alpha)
    prod_pipeline.fit(df['clean_text'], df['label'])

    # 5. Serialize pipeline
    joblib.dump(prod_pipeline, args.model_path)
    file_size_kb = os.path.getsize(args.model_path) / 1024
    print(f"[Production] Successfully saved model to '{args.model_path}' ({file_size_kb:.1f} KB).")

    # 6. Sanity check verification
    print("\n[Sanity Check] Verifying serialized model inference:")
    sample_tests = [
        "URGENT! You have won a £1,000 cash prize! Text CLAIM to 88010 now.",
        "Hey, are we still meeting for lunch today at 1pm?"
    ]
    reloaded_model = joblib.load(args.model_path)
    for sample in sample_tests:
        pred = reloaded_model.predict([normalize_text(sample)])[0]
        prob = reloaded_model.predict_proba([normalize_text(sample)])[0]
        spam_p = prob[list(reloaded_model.classes_).index('spam')] * 100
        print(f"  - \"{sample[:40]}...\" -> [{pred.upper()}] (Spam Confidence: {spam_p:.1f}%)")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
