# scripts/visuals.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, ConfusionMatrixDisplay


def plot_anomaly_counts(df, save_path="outputs/anomaly_barplot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = {
        "Isolation Forest": "anomaly_label",
        "LOF": "lof_label",
        "One-Class SVM": "svm_label"
    }

    counts = {"Model": [], "Anomalous": [], "Normal": []}

    for model_name, column in models.items():
        if column in df.columns:
            vc = df[column].value_counts()
            counts["Model"].append(model_name)
            counts["Anomalous"].append(vc.get("Anomalous", 0))
            counts["Normal"].append(vc.get("Normal", 0))

    count_df = pd.DataFrame(counts)
    count_df.set_index("Model")[["Anomalous", "Normal"]].plot(
        kind="bar", stacked=True, colormap="Set2", figsize=(8, 5))
    plt.title("Anomalies Detected by Each Model")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_user_feature_distribution(df, save_path="outputs/feature_boxplots.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feature_df = df[[
        "login_frequency",
        "failed_login_attempts",
        "file_access_count",
        "application_activity_count",
        "avg_session_duration",
        "total_event_count"
    ]].drop_duplicates()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=feature_df)
    plt.xticks(rotation=45)
    plt.title("User Behavior Feature Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_heatmap(df, save_path="outputs/user_feature_correlation.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    feature_df = df[[
        "login_frequency",
        "failed_login_attempts",
        "file_access_count",
        "application_activity_count",
        "avg_session_duration",
        "total_event_count"
    ]].drop_duplicates()

    plt.figure(figsize=(10, 6))
    sns.heatmap(feature_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")


# 🔹 NEW: Confusion Matrix for RF & SVM classification

def plot_confusion_matrix(model, X_test, y_test, label_encoder, model_name, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_supervised_metrics(models, X_test, y_test, label_encoder, save_path="outputs/supervised_metrics.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scores = {"Model": [], "Accuracy": [], "Precision": [], "F1 Score": []}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        scores["Model"].append(name)
        scores["Accuracy"].append(accuracy_score(y_test, y_pred))
        scores["Precision"].append(precision_score(y_test, y_pred, average="weighted"))
        scores["F1 Score"].append(f1_score(y_test, y_pred, average="weighted"))

    score_df = pd.DataFrame(scores).set_index("Model")

    score_df.plot(kind="bar", figsize=(10, 5), ylim=(0.8, 1.0))
    plt.title("Supervised Model Performance Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")
