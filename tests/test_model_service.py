"""
Unit tests for ChurnLens model loading, feature encoding, and prediction logic.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add repo root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model_service import (
    load_model,
    predict_single,
    predict_batch,
    encode_features,
    assess_risk_factors,
    FEATURE_COLUMNS,
)


def test_load_model():
    """Verify model artifact can be found, loaded, and has expected attributes."""
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert len(model.feature_names_in_) == 19


def test_encode_features():
    """Verify input dictionary encoding outputs DataFrame with exact model feature schema."""
    sample_input = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
    }
    encoded = encode_features(sample_input)
    assert isinstance(encoded, pd.DataFrame)
    assert list(encoded.columns) == FEATURE_COLUMNS
    assert len(encoded) == 1
    assert encoded["gender"].iloc[0] == 0
    assert encoded["Contract"].iloc[0] == 0
    assert encoded["tenure"].iloc[0] == 12.0


def test_predict_single_high_risk():
    """Verify high-risk customer profile yields high churn probability and HIGH risk level."""
    high_risk_input = {
        "gender": "Male",
        "SeniorCitizen": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 95.0,
    }
    res = predict_single(high_risk_input)
    assert "prediction" in res
    assert "churn_probability" in res
    assert "stay_probability" in res
    assert "risk_level" in res
    assert "risk_drivers" in res
    assert "retention_recommendations" in res

    assert res["prediction"] == 1
    assert res["churn_probability"] >= 70.0
    assert res["risk_level"] == "HIGH"
    assert pytest.approx(res["churn_probability"] + res["stay_probability"], abs=0.1) == 100.0
    assert len(res["risk_drivers"]) > 0
    assert len(res["retention_recommendations"]) > 0


def test_predict_single_low_risk():
    """Verify low-risk customer profile yields low churn probability and LOW risk level."""
    low_risk_input = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 48.0,
        "TotalCharges": 3264.0,
    }
    res = predict_single(low_risk_input)
    assert res["prediction"] == 0
    assert res["churn_probability"] < 40.0
    assert res["risk_level"] == "LOW"


def test_predict_batch():
    """Verify batch prediction processes DataFrame and generates correct output columns."""
    df_sample = pd.DataFrame([
        {
            "Contract": "Month-to-month",
            "tenure": 2,
            "MonthlyCharges": 90.0,
            "TotalCharges": 180.0,
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
        },
        {
            "Contract": "Two year",
            "tenure": 60,
            "MonthlyCharges": 40.0,
            "TotalCharges": 2400.0,
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "DSL",
        },
    ])
    scored = predict_batch(df_sample)
    assert "Churn_Prediction" in scored.columns
    assert "Churn_Probability_%" in scored.columns
    assert "Risk_Tier" in scored.columns
    assert len(scored) == 2
    assert scored["Risk_Tier"].iloc[0] in ["HIGH", "MEDIUM"]
    assert scored["Risk_Tier"].iloc[1] == "LOW"
