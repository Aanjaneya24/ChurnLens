# 🔮 ChurnLens — Customer Churn Prediction & Business Intelligence

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://render.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An enterprise-ready **Customer Churn Prediction and Business Intelligence Platform** built on the **Telco Customer Churn Dataset** (7,032 subscriber records).

ChurnLens combines machine learning risk forecasting, interactive business intelligence, automated retention playbooks, bulk CSV scoring, and an embedded SQL analytics studio into a single application.

---

## 📌 Executive Summary

Customer churn directly erodes recurring revenue. In telecom and subscription businesses, acquiring new customers costs **5x to 7x more** than retaining existing ones. 

**ChurnLens** bridges the gap between predictive data science and business operations:
1. **Predictive AI Engine**: Scores customer churn probability with **78.9% accuracy** and an **ROC-AUC of 0.825**.
2. **Explainable Risk Drivers**: Automatically pinpoints *why* an account is at risk (contract type, tenure cohort, billing quartile, unbundled services).
3. **Actionable Retention Playbooks**: Delivers targeted commercial recommendations (contract discounts, auto-pay credits, tech support trials) before cancellation occurs.
4. **Bulk Scoring Pipeline**: Upload customer CSV files to batch-score accounts and prioritize retention outreach by risk tier (High, Medium, Low).
5. **Interactive SQL Studio**: Run SQL queries live against the customer database in-memory without external infrastructure dependencies.

---

## 🏗️ Architecture & Tech Stack

```text
ChurnLens Platform
│
├── 🔮 UI & BI Layer           ── Streamlit Web Application (app/app.py)
│                                 • Interactive Single Customer Scoring
│                                 • Batch CSV Scoring & Export
│                                 • Executive KPI & Trend Visualizations
│                                 • SQL Analytics Studio (in-memory SQLite)
│                                 • Model Card & Governance View
│
├── 🤖 Machine Learning Layer   ── Random Forest Classifier (100 estimators)
│                                 • app/model_service.py (Modular inference)
│                                 • scripts/train.py (Reproducible training pipeline)
│                                 • app/churn_model.pkl (Compressed 1.7 MB artifact)
│
├── 🗄️ Analytics & SQL Layer   ── SQL Analytics (sql/queries.sql)
│                                 • scripts/run_sql_analysis.py (SQLite BI runner)
│                                 • Contract, tenure, payment, and revenue queries
│
└── 🧪 Quality & Deployment    ── Automated Tests & Render Blueprint
                                  • tests/ (pytest suite for model, SQL, & data)
                                  • render.yaml (One-click Render deployment)
```

### Technology Highlights
- **Language**: Python 3.11+
- **Machine Learning**: `scikit-learn`, `joblib`, `numpy`, `pandas`
- **Interactive UI**: `streamlit`, `matplotlib`, `seaborn`
- **Database Engine**: In-memory `sqlite3` (with optional `sqlalchemy` / `pymysql` connectivity)
- **Quality Assurance**: `pytest`
- **Hosting / PaaS**: Render Web Service (`render.yaml`)

---

## 📂 Project Directory Structure

```text
ChurnLens/
├── .streamlit/
│   └── config.toml             # Streamlit server and branding theme configuration
├── app/
│   ├── app.py                  # Main Streamlit web platform & dashboard
│   ├── churn_model.pkl         # Trained, compressed Random Forest model (~1.7 MB)
│   └── model_service.py        # Inference engine, feature encoder, & risk diagnostics
├── data/
│   ├── cleaned/
│   │   └── churn_cleaned.csv   # Cleaned dataset (7,032 rows, 0 nulls, encoded target)
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original Kaggle raw dataset
│   ├── visuals/                # Model evaluation & EDA visual artifacts
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── ...
│   └── model_metrics.json      # Serialized test-set evaluation scores
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis & visual discovery
│   └── model.ipynb             # Algorithm evaluation, feature rankings, & tuning
├── powerbi/
│   └── churn_dashboard.pbix    # Original Power BI executive dashboard
├── scripts/
│   ├── run_sql_analysis.py     # Standalone runner for all SQL business queries
│   └── train.py                # End-to-end model training, evaluation, & export script
├── sql/
│   └── queries.sql             # Cross-engine SQL analytics queries (SQLite/MySQL/Postgres)
├── tests/
│   ├── test_data_integrity.py  # Tests for dataset row count, nulls, and schema
│   ├── test_model_service.py   # Tests for model loading, encoding, and risk tiers
│   └── test_sql_queries.py     # Tests for SQL syntax and query execution
├── .env.example                # Template for optional database/server environment variables
├── .gitignore                  # Comprehensive Python, IDE, and OS exclusion patterns
├── LICENSE                     # MIT License with original attribution
├── README.md                   # Complete documentation
├── render.yaml                 # Render cloud deployment specification
└── requirements.txt            # Tested, pinned dependency requirements
```

---

## 📊 Key Business Intelligence Insights

Analysis across 7,032 customer accounts reveals critical risk drivers and revenue exposure:

| Business Dimension | High-Risk Segment | Churn Rate | Low-Risk Segment | Churn Rate | Actionable Retention Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Contract Type** | Month-to-Month | **42.71%** | Two-Year Contract | **2.85%** | 15% discount incentive for switching to annual commitments |
| **Tenure Lifecycle** | 0–1 Year (Onboarding) | **47.68%** | 4+ Years (Loyal) | **9.51%** | 30/60/90-day proactive customer success check-ins |
| **Payment Channel** | Electronic Check | **45.29%** | Credit Card (Auto) | **15.25%** | $5/month statement credit for enrolling in Auto-Pay |
| **Service Bundling** | Fiber Optic (No Tech Support) | **49.37%** | Fiber Optic (With Tech Support) | **22.63%** | Complimentary 90-day trial of cybersecurity & support bundle |
| **Revenue Impact** | High Monthly Billing (≥$70) | **52.80%** | Total Churned Revenue | **$139,130/mo** | Focus retention campaigns on top billing quartile |

---

## 🤖 Machine Learning Model Performance

The ChurnLens model is an optimized **Random Forest Classifier** trained on an 80/20 stratified holdout split:

| Evaluation Metric | Holdout Test Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **78.89%** | Overall correct predictions |
| **ROC-AUC** | **0.8248** | Separability between churners and retained accounts |
| **Precision (Churn)** | **0.6288** | Minimizes false alarms in outreach campaigns |
| **Recall (Churn)** | **0.5027** | Captures over half of all churners prior to cancellation |
| **F1-Score** | **0.5587** | Balanced harmonic mean |

### Top Predictive Features (Gini Importance)
1. **TotalCharges** — Cumulative account billing
2. **MonthlyCharges** — Current plan cost
3. **tenure** — Duration of customer relationship
4. **Contract** — Commitment structure (Month-to-month vs Annual)
5. **PaymentMethod** — Payment vehicle (Electronic check vs Automated ACH/Card)
6. **OnlineSecurity & TechSupport** — Value-added protective add-ons

---

## 🚀 Local Installation & Quickstart

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 1. Clone or Open the Repository
```bash
git clone https://github.com/Aanjaneya24/ChurnLens.git
cd ChurnLens
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite
```bash
pytest -v
```
All 9 unit and integration tests should pass (`tests/test_model_service.py`, `tests/test_sql_queries.py`, `tests/test_data_integrity.py`).

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ CLI Utilities & Model Retraining

### Retrain the Machine Learning Model
To retrain the model on the latest data, regenerate evaluation metrics, and update confusion matrix / feature importance plots:
```bash
python scripts/train.py
```

### Execute the SQL Business Intelligence Suite
To run all 7 analytical queries against the in-memory SQLite database and print structured tables:
```bash
python scripts/run_sql_analysis.py
```

---

## ☁️ Deployment on Render

ChurnLens includes a ready-to-use [`render.yaml`](render.yaml) blueprint specification for zero-configuration deployment on [Render](https://render.com).

### Method A: One-Click Blueprint Deployment (Recommended)
1. Push your repository to your GitHub account (see instructions below).
2. Log in to [dashboard.render.com](https://dashboard.render.com).
3. Click **New +** → **Blueprint**.
4. Connect your GitHub repository (`ChurnLens`).
5. Render automatically reads `render.yaml` and provisions the Web Service with the correct build and start commands.
6. Click **Apply**. Your app will be live with an HTTPS URL.

### Method B: Manual Web Service Configuration
If setting up manually in the Render dashboard:
- **Service Type**: Web Service
- **Name**: `churnlens`
- **Runtime**: `Python 3`
- **Branch**: `main`
- **Build Command**:
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
  ```
- **Health Check Path**: `/_stcore/health`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.11.9`
  - `STREAMLIT_SERVER_HEADLESS`: `true`
  - `STREAMLIT_SERVER_ENABLE_CORS`: `false`
  - `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION`: `false`

---

## 🐙 Publishing to Your GitHub Account

To establish this project under your own GitHub profile (`ChurnLens`):

1. **Create an empty repository on GitHub**:
   - Go to [github.com/new](https://github.com/new).
   - Set Repository Name to: `ChurnLens`
   - Description: `ChurnLens — Customer Churn Prediction & Business Intelligence Platform`
   - Visibility: Public or Private (your choice)
   - Do **NOT** initialize with a README, .gitignore, or license (we already have them).

2. **Set Remote and Push from your terminal**:
   ```bash
   cd /path/to/ChurnLens

   # Point origin to your new repository
   git remote set-url origin https://github.com/Aanjaneya24/ChurnLens.git

   # Verify remote URL
   git remote -v

   # Stage all updated ChurnLens files
   git add .

   # Commit changes
   git commit -m "Transform project into ChurnLens: BI platform, modular service, tests, and Render deployment"

   # Push to main branch
   git branch -M main
   git push -u origin main
   ```

---

## 👤 Author & Maintainer

**Aanjaneya Pandey**  
- **GitHub**: [@Aanjaneya24](https://github.com/Aanjaneya24)  
- **Repository**: [https://github.com/Aanjaneya24/ChurnLens](https://github.com/Aanjaneya24/ChurnLens)

---

## ⚖️ License

Distributed under the **MIT License**. Copyright (c) 2026 Aanjaneya Pandey. See [`LICENSE`](LICENSE) for complete terms.
