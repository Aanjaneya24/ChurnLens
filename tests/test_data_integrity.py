"""
Data integrity tests for Telco Customer Churn datasets.
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_PATH = ROOT_DIR / "data" / "cleaned" / "churn_cleaned.csv"


def test_cleaned_dataset_integrity():
    assert CLEANED_PATH.exists()
    df = pd.read_csv(CLEANED_PATH)

    # Cleaned dataset must have 7,032 rows (7,043 - 11 blank TotalCharges)
    assert len(df) == 7032
    assert "TotalCharges" in df.columns
    assert df["TotalCharges"].isnull().sum() == 0
    assert "Churn" in df.columns
    assert set(df["Churn"].unique()).issubset({0, 1})
    assert "customerID" in df.columns


def test_raw_dataset_exists():
    assert RAW_PATH.exists()
    df = pd.read_csv(RAW_PATH)
    assert len(df) == 7043
