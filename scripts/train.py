"""
ChurnLens Model Training and Evaluation Pipeline.
Trains a Random Forest classifier on the cleaned Telco Customer Churn dataset,
evaluates performance metrics, and exports the serialized model artifact.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned" / "churn_cleaned.csv"
MODEL_PATH = BASE_DIR / "app" / "churn_model.pkl"
VISUALS_DIR = BASE_DIR / "data" / "visuals"
METRICS_PATH = BASE_DIR / "data" / "model_metrics.json"


def load_and_preprocess(data_path: Path):
    """Load cleaned data and encode categorical features identically to the inference pipeline."""
    df = pd.read_csv(data_path)

    # Ensure numeric columns
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Encode categorical columns
    le_dict = {}
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if "customerID" in cat_cols:
        cat_cols.remove("customerID")

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = list(le.classes_)

    X = df.drop(columns=["customerID", "Churn"], errors="ignore")
    y = df["Churn"].astype(int)

    return X, y, le_dict


def train_and_evaluate():
    print(f"Loading data from {DATA_PATH}...")
    X, y, le_dict = load_and_preprocess(DATA_PATH)
    print(f"Dataset shape: {X.shape}, Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Train Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        max_depth=12,
        min_samples_split=5,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "features": list(X.columns),
    }

    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and metrics
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, MODEL_PATH, compress=3)
    print(f"Model saved to {MODEL_PATH} (compressed size: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB)")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # Generate and save visuals
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
    )
    plt.title("ChurnLens — Random Forest Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    feat_imp.tail(10).plot(kind="barh", color="#2563EB")
    plt.title("ChurnLens — Top 10 Most Predictive Features")
    plt.xlabel("Feature Importance (Gini)")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png", dpi=200)
    plt.close()

    print(f"Updated visuals saved to {VISUALS_DIR}")
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
