# run_pipeline.py

import pandas as pd
import os
import joblib
from scripts.preprocessing import load_and_clean
from scripts.feature_engineering import compute_features as engineer_features
from scripts.modeling import run_unsupervised_models, train_supervised_model as run_supervised_models
from scripts.visuals import (
    plot_anomaly_counts,
    plot_user_feature_distribution,
    plot_heatmap,
    plot_confusion_matrix,
    plot_supervised_metrics
)
from sklearn.model_selection import train_test_split

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n🚀 Running User Behaviour Analysis Pipeline...\n")

    # STEP 1: Preprocessing (drop columns, fix timestamps, merge categories, fill users)
    print("📥 Step 1: Load and Clean Data")
    df = load_and_clean(
        raw_log_path="data/raw_logs.csv",
        category_map_path="data/category_mapping.csv",
        save_path="data/Windows_Events_Log_With_Synthetic_Users.csv"
    )

    # STEP 2: Feature Engineering
    print("🧠 Step 2: Feature Engineering")
    df, user_summary = engineer_features(df)

    # Save intermediate feature-enhanced dataset
    df.to_csv("data/Windows_Event_Log_Feature_Enhanced.csv", index=False)

    # STEP 3: Anomaly Detection
    print("🤖 Step 3: Run Anomaly Detection Models")
    df_unsupervised = run_unsupervised_models(df)
    df = df.merge(df_unsupervised, on="Security Identifier", how="left")

    # STEP 4: Supervised Model
    rf_model, svm_model, vectorizer, label_encoder, df = run_supervised_models(df)

    # Save final result
    df.to_csv("data/Windows_Event_Log_Feature_Enhanced.csv", index=False)
    print("✅ Saved enhanced dataset with anomalies to: data/Windows_Event_Log_Feature_Enhanced.csv")

    # STEP 5: Visualizations
    print("\n📊 Step 4: Generating Visualizations")
    plot_anomaly_counts(df)
    plot_user_feature_distribution(user_summary)
    plot_heatmap(user_summary)

    # Metrics & Confusion Matrices
    X_all = vectorizer.transform(df["processed_text"])
    y_all = df["level_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    plot_confusion_matrix(rf_model, X_test, y_test, label_encoder, "Random Forest", "outputs/rf_confusion_matrix.png")
    plot_confusion_matrix(svm_model, X_test, y_test, label_encoder, "SVM", "outputs/svm_confusion_matrix.png")
    plot_supervised_metrics({"Random Forest": rf_model, "SVM": svm_model}, X_test, y_test, label_encoder)

    print("\n🎉 Pipeline Complete! All models and visuals saved in 'models' and 'outputs' folders.")

if __name__ == "__main__":
    main()
