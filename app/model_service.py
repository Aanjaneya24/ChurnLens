"""
ChurnLens Model Service.
Encapsulates model loading, feature encoding, single-customer inference,
risk-factor analysis, retention recommendations, and batch prediction.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

# Feature schema required by the model
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Explicit mappings matching training LabelEncoder behavior
ENCODING_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "SeniorCitizen": {"No": 0, "Yes": 1, 0: 0, 1: 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    },
}

_MODEL_CACHE = None


def get_model_path() -> Path:
    """Resolve model path reliably regardless of current working directory."""
    candidates = [
        Path(__file__).resolve().parent / "churn_model.pkl",
        Path.cwd() / "app" / "churn_model.pkl",
        Path.cwd() / "churn_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_model():
    """Load and cache the trained model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        model_path = get_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please run `python scripts/train.py` first."
            )
        _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def encode_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Encode raw user input dictionary into the numerical DataFrame expected by the model.
    Handles defaults and data type conversions safely.
    """
    encoded = {}

    for col in FEATURE_COLUMNS:
        val = input_data.get(col)
        if col in ENCODING_MAPS:
            mapping = ENCODING_MAPS[col]
            if val in mapping:
                encoded[col] = mapping[val]
            elif str(val).capitalize() in mapping:
                encoded[col] = mapping[str(val).capitalize()]
            else:
                # Default fallback to 0
                encoded[col] = 0
        else:
            # Numerical columns
            try:
                encoded[col] = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                encoded[col] = 0.0

    return pd.DataFrame([encoded], columns=FEATURE_COLUMNS)


def assess_risk_factors(raw_data: Dict[str, Any], churn_prob: float) -> Tuple[List[Dict[str, str]], List[str]]:
    """Generate explainable business risk drivers and actionable retention strategies."""
    drivers = []
    actions = []

    contract = raw_data.get("Contract", "Month-to-month")
    tenure = float(raw_data.get("tenure", 0))
    monthly = float(raw_data.get("MonthlyCharges", 0))
    payment = raw_data.get("PaymentMethod", "")
    tech_support = raw_data.get("TechSupport", "No")
    online_sec = raw_data.get("OnlineSecurity", "No")
    internet = raw_data.get("InternetService", "")

    # Contract Risk
    if contract == "Month-to-month":
        drivers.append({
            "category": "Contract",
            "impact": "High",
            "detail": "Month-to-month contracts have a 42.7% baseline churn rate.",
        })
        actions.append("Offer a 15% discount incentive for enrolling in a 1-year or 2-year commitment.")
    elif contract in ["One year", "Two year"]:
        drivers.append({
            "category": "Contract",
            "impact": "Low (Protective)",
            "detail": f"{contract} contract provides strong retention stability.",
        })

    # Tenure Risk
    if tenure <= 12:
        drivers.append({
            "category": "Tenure",
            "impact": "High",
            "detail": f"Customer is in first-year lifecycle ({int(tenure)} mos). Onboarding period is highest risk.",
        })
        actions.append("Initiate proactive customer success check-in and satisfaction review.")
    elif tenure > 36:
        drivers.append({
            "category": "Tenure",
            "impact": "Low (Protective)",
            "detail": f"High tenure loyalty ({int(tenure)} months active).",
        })

    # Pricing Risk
    if monthly >= 75.0:
        drivers.append({
            "category": "Pricing",
            "impact": "Medium-High",
            "detail": f"Monthly charges of ${monthly:.2f} are in the top billing quartile.",
        })
        actions.append("Review billing plan for unused services or bundle optimizations to enhance perceived value.")

    # Payment Method Risk
    if payment == "Electronic check":
        drivers.append({
            "category": "Payment",
            "impact": "Medium",
            "detail": "Electronic check users churn at more than double the rate of automated payment methods.",
        })
        actions.append("Offer a $5/month statement credit for switching to automated Credit Card or ACH transfer.")

    # Service Protection Deficit
    if internet in ["Fiber optic", "DSL"]:
        if tech_support == "No" or online_sec == "No":
            drivers.append({
                "category": "Services",
                "impact": "Medium",
                "detail": "Unprotected internet package (lacks Tech Support or Online Security add-on).",
            })
            actions.append("Offer complimentary 3-month trial of Premium Tech Support & Cybersecurity shield.")

    if not actions:
        actions.append("Maintain standard relationship cadence; customer profile shows healthy engagement.")

    return drivers, actions


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run churn prediction on a single customer dictionary.
    Returns prediction label, probabilities, risk tier, drivers, and recommendations.
    """
    model = load_model()
    encoded_df = encode_features(input_data)

    pred = int(model.predict(encoded_df)[0])
    probs = model.predict_proba(encoded_df)[0]
    churn_prob = float(probs[1])
    stay_prob = float(probs[0])

    if churn_prob >= 0.70:
        risk_level = "HIGH"
        risk_color = "red"
    elif churn_prob >= 0.40:
        risk_level = "MEDIUM"
        risk_color = "orange"
    else:
        risk_level = "LOW"
        risk_color = "green"

    drivers, recommendations = assess_risk_factors(input_data, churn_prob)

    return {
        "prediction": pred,
        "prediction_label": "Likely to Churn" if pred == 1 else "Likely to Stay",
        "churn_probability": round(churn_prob * 100, 2),
        "stay_probability": round(stay_prob * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_drivers": drivers,
        "retention_recommendations": recommendations,
        "features_used": encoded_df.to_dict(orient="records")[0],
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch DataFrame of customer records.
    Returns original DataFrame with predicted churn probability and risk tier.
    """
    model = load_model()
    result_df = df.copy()

    # Encode all rows
    encoded_rows = []
    for _, row in df.iterrows():
        encoded_df = encode_features(row.to_dict())
        encoded_rows.append(encoded_df.iloc[0])

    X_batch = pd.DataFrame(encoded_rows)
    probs = model.predict_proba(X_batch)[:, 1]
    preds = model.predict(X_batch)

    result_df["Churn_Prediction"] = np.where(preds == 1, "Likely to Churn", "Likely to Stay")
    result_df["Churn_Probability_%"] = (probs * 100).round(2)
    result_df["Risk_Tier"] = pd.cut(
        result_df["Churn_Probability_%"],
        bins=[-np.inf, 39.99, 69.99, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    return result_df
