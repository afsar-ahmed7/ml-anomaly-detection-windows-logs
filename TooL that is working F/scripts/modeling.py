import pandas as pd
import numpy as np
import re
import joblib

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download("punkt")
nltk.download("stopwords")

# ---------------- TEXT PREPROCESSING ----------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

# ---------------- UNSUPERVISED MODELS ----------------

def run_unsupervised_models(df):
    """
    Run Isolation Forest, LOF, and One-Class SVM on user behavior features.
    Saves Isolation Forest and One-Class SVM models to /models folder.
    Returns updated dataframe with anomaly labels.
    """
    import os
    df["avg_session_duration"] = df["avg_session_duration"].fillna(0)

    user_df = df[[
        "Security Identifier",
        "login_frequency",
        "failed_login_attempts",
        "file_access_count",
        "application_activity_count",
        "avg_session_duration",
        "total_event_count"
    ]].drop_duplicates().reset_index(drop=True)

    user_ids = user_df["Security Identifier"]
    X = user_df.drop("Security Identifier", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Isolation Forest ---
    iso_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    user_df["anomaly_label"] = iso_model.fit_predict(X_scaled)
    user_df["anomaly_label"] = user_df["anomaly_label"].map({1: "Normal", -1: "Anomalous"})

    # --- Local Outlier Factor ---
    lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
    user_df["lof_label"] = lof.fit_predict(X_scaled)
    user_df["lof_label"] = user_df["lof_label"].map({1: "Normal", -1: "Anomalous"})

    # --- One-Class SVM ---
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    user_df["svm_label"] = oc_svm.fit_predict(X_scaled)
    user_df["svm_label"] = user_df["svm_label"].map({1: "Normal", -1: "Anomalous"})

    user_df["Security Identifier"] = user_ids

    # ✅ Save Isolation Forest and One-Class SVM models
    os.makedirs("models", exist_ok=True)
    joblib.dump(iso_model, "models/isolation_forest.pkl")
    joblib.dump(oc_svm, "models/oneclass_svm.pkl")
    print("✅ Saved: isolation_forest.pkl and oneclass_svm.pkl to models/")

    return user_df

# ---------------- SUPERVISED CLASSIFICATION ----------------
def train_supervised_model(df, save=True):
    """
    Train a text classification model to predict 'Level' using TF-IDF + RF/SVM.
    Returns trained models and vectorizer.
    """

    df = df.dropna(subset=["Level", "Event Description Summary"]).copy()
    df["processed_text"] = df["Event Description Summary"].apply(preprocess_text)

    # Encode labels
    label_encoder = LabelEncoder()
    df["level_encoded"] = label_encoder.fit_transform(df["Level"])

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["processed_text"])
    y = df["level_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Random Forest ---
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("\n📊 Random Forest:")
    evaluate_model(rf_model, X_test, y_test, label_encoder)

    # --- SVM ---
    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train, y_train)
    print("\n📊 SVM:")
    evaluate_model(svm_model, X_test, y_test, label_encoder)

    if save:
        joblib.dump(rf_model, "rf_model_level_classifier.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        joblib.dump(label_encoder, "level_label_encoder.pkl")
        print("✅ Saved: model, vectorizer, and label encoder")

    return rf_model, svm_model, vectorizer, label_encoder, df

# ---------------- METRICS ----------------
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
    print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
