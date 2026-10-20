"""
ChurnLens — Customer Churn Prediction & Business Intelligence Platform.
A unified machine learning and business intelligence application for subscription retention.
"""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Import ChurnLens model service
try:
    from app.model_service import (
        predict_single,
        predict_batch,
        load_model,
        FEATURE_COLUMNS,
    )
except ImportError:
    from model_service import (
        predict_single,
        predict_batch,
        load_model,
        FEATURE_COLUMNS,
    )

# ── Paths ─────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "cleaned" / "churn_cleaned.csv"
VISUALS_DIR = ROOT_DIR / "data" / "visuals"
SQL_PATH = ROOT_DIR / "sql" / "queries.sql"

# ── Streamlit Page Configuration ──────────────────────────────────
st.set_page_config(
    page_title="ChurnLens — Churn Prediction & BI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Modern Styling ─────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #E0E7FF !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .risk-badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .risk-badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_cleaned_data():
    """Load and cache the baseline dataset for analytics."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()



@st.cache_resource
def get_db_connection():
    """Create and cache an in-memory SQLite database populated with customer dataset.
    Uses @st.cache_resource because sqlite3.Connection is not pickle-serializable.
    """
    df = get_cleaned_data()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    if not df.empty:
        df.to_sql("customers", conn, index=False)
    return conn



# ── Sidebar Navigation ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullseye.png", width=64)
    st.title("ChurnLens")
    st.caption("Customer Churn Prediction & Business Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🔮 Single Customer Prediction",
            "📁 Batch Scoring & Intelligence",
            "📊 Executive BI Dashboard",
            "🗄️ SQL Analytics Studio",
            "🤖 Model Governance & Card",
            "ℹ️ About & Attribution",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    try:
        load_model()
        st.success("🟢 ML Engine: Online (Random Forest)")
    except Exception as e:
        st.error(f"🔴 ML Engine: Offline ({e})")

    st.markdown(
        """
        <small style='color: #64748B;'>
        <strong>ChurnLens Platform v2.0</strong><br>
        Developed & Maintained by Aanjaneya Pandey.<br>
        Open-Source MIT License.
        </small>
        """,
        unsafe_allow_html=True,
    )

# ── Main Header Banner ─────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔮 ChurnLens</h1>
        <p>AI-Powered Customer Retention Forecasting, Business Intelligence & Risk Playbooks</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════
# 1. SINGLE CUSTOMER PREDICTION
# ════════════════════════════════════════════════════════════════════
if page == "🔮 Single Customer Prediction":
    st.subheader("🎯 Real-Time Customer Risk Scoring")
    st.write(
        "Enter subscriber attributes below to predict churn probability, evaluate risk drivers, and retrieve custom retention strategies."
    )

    with st.expander("💡 Quick Fill Presets (Click to load scenario)", expanded=False):
        preset_cols = st.columns(3)
        if preset_cols[0].button("⚠️ High Churn Risk Profile"):
            st.session_state["p_tenure"] = 2
            st.session_state["p_contract"] = "Month-to-month"
            st.session_state["p_monthly"] = 92.0
            st.session_state["p_payment"] = "Electronic check"
            st.session_state["p_internet"] = "Fiber optic"
            st.session_state["p_security"] = "No"
            st.session_state["p_support"] = "No"
            st.session_state["p_paperless"] = "Yes"
        if preset_cols[1].button("✅ High Loyalty Profile"):
            st.session_state["p_tenure"] = 62
            st.session_state["p_contract"] = "Two year"
            st.session_state["p_monthly"] = 52.0
            st.session_state["p_payment"] = "Credit card (automatic)"
            st.session_state["p_internet"] = "DSL"
            st.session_state["p_security"] = "Yes"
            st.session_state["p_support"] = "Yes"
            st.session_state["p_paperless"] = "No"
        if preset_cols[2].button("🔄 Typical Average Customer"):
            st.session_state["p_tenure"] = 24
            st.session_state["p_contract"] = "One year"
            st.session_state["p_monthly"] = 65.0
            st.session_state["p_payment"] = "Bank transfer (automatic)"
            st.session_state["p_internet"] = "Fiber optic"
            st.session_state["p_security"] = "Yes"
            st.session_state["p_support"] = "Yes"
            st.session_state["p_paperless"] = "Yes"

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with col2:
            st.markdown("#### 📦 Subscribed Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            if phone == "No":
                multiple = "No phone service"
                st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
            else:
                multiple = st.selectbox("Multiple Lines", ["No", "Yes"])

            internet = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(st.session_state.get("p_internet", "Fiber optic")),
            )

            if internet == "No":
                st.info("ℹ️ Internet add-ons auto-set to 'No internet service'")
                security = "No internet service"
                backup = "No internet service"
                protection = "No internet service"
                support = "No internet service"
                tv = "No internet service"
                movies = "No internet service"
            else:
                sec_default = st.session_state.get("p_security", "No")
                sup_default = st.session_state.get("p_support", "No")
                security = st.selectbox("Online Security", ["No", "Yes"], index=1 if sec_default == "Yes" else 0)
                backup = st.selectbox("Online Backup", ["No", "Yes"])
                protection = st.selectbox("Device Protection", ["No", "Yes"])
                support = st.selectbox("Tech Support", ["No", "Yes"], index=1 if sup_default == "Yes" else 0)
                tv = st.selectbox("Streaming TV", ["No", "Yes"])
                movies = st.selectbox("Streaming Movies", ["No", "Yes"])

        with col3:
            st.markdown("#### 💳 Account & Billing")
            tenure = st.slider(
                "Tenure (months active)",
                min_value=0,
                max_value=72,
                value=int(st.session_state.get("p_tenure", 12)),
            )

            contract_list = ["Month-to-month", "One year", "Two year"]
            contract_default = st.session_state.get("p_contract", "Month-to-month")
            contract = st.selectbox(
                "Contract Type",
                contract_list,
                index=contract_list.index(contract_default) if contract_default in contract_list else 0,
            )

            billing_default = st.session_state.get("p_paperless", "Yes")
            billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=0 if billing_default == "Yes" else 1,
            )

            payment_list = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            payment_default = st.session_state.get("p_payment", "Electronic check")
            payment = st.selectbox(
                "Payment Method",
                payment_list,
                index=payment_list.index(payment_default) if payment_default in payment_list else 0,
            )

            monthly = st.slider(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=120.0,
                value=float(st.session_state.get("p_monthly", 65.0)),
                step=0.5,
            )

            suggested_total = round(float(tenure * monthly), 2)
            total = st.number_input(
                "Total Charges ($) [Estimated: tenure × monthly]",
                min_value=0.0,
                max_value=15000.0,
                value=suggested_total,
                step=10.0,
            )

        submit_btn = st.form_submit_button("🔮 Calculate Churn Risk Score", use_container_width=True)

    if submit_btn:
        input_payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": billing,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        with st.spinner("Scoring customer profile..."):
            result = predict_single(input_payload)

        st.markdown("---")
        st.subheader("🎯 Prediction & Risk Diagnostic")

        # Top summary cards
        c_res1, c_res2, c_res3 = st.columns([1.5, 1, 1])

        with c_res1:
            if result["risk_level"] == "HIGH":
                st.error(
                    f"### ⚠️ HIGH RISK — {result['prediction_label'].upper()}\n"
                    f"This customer displays significant signs of imminent churn. Immediate retention intervention is recommended."
                )
            elif result["risk_level"] == "MEDIUM":
                st.warning(
                    f"### 🟡 MEDIUM RISK — {result['prediction_label'].upper()}\n"
                    f"Customer displays vulnerable metrics. Proactive engagement will mitigate risk."
                )
            else:
                st.success(
                    f"### ✅ LOW RISK — {result['prediction_label'].upper()}\n"
                    f"Customer profile indicates high loyalty and low propensity to cancel."
                )

        with c_res2:
            st.metric(
                label="Churn Probability",
                value=f"{result['churn_probability']}%",
                delta=f"{result['churn_probability'] - 26.58:+.1f}% vs baseline",
                delta_color="inverse",
            )

        with c_res3:
            st.metric(
                label="Stay (Retention) Probability",
                value=f"{result['stay_probability']}%",
            )

        # Risk drivers and retention actions
        col_dr, col_act = st.columns(2)

        with col_dr:
            st.markdown("#### 🔍 Primary Risk Drivers")
            if result["risk_drivers"]:
                for d in result["risk_drivers"]:
                    st.markdown(f"- **[{d['category']} — {d['impact']}]**: {d['detail']}")
            else:
                st.write("No adverse risk flags identified for this profile.")

        with col_act:
            st.markdown("#### 📋 Recommended Retention Playbook")
            if result["retention_recommendations"]:
                for act in result["retention_recommendations"]:
                    st.markdown(f"- 💡 {act}")


# ════════════════════════════════════════════════════════════════════
# 2. BATCH SCORING & INTELLIGENCE
# ════════════════════════════════════════════════════════════════════
elif page == "📁 Batch Scoring & Intelligence":
    st.subheader("📁 Bulk Customer Churn Scoring")
    st.write(
        "Upload a CSV of customer records to score churn propensity in bulk, segment high-risk accounts, and download retention action lists."
    )

    col_up, col_sm = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    with col_sm:
        st.write("Or evaluate sample records:")
        use_sample = st.button("📥 Load 25 Sample Customers")

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {len(batch_df)} customer records.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample:
        full_df = get_cleaned_data()
        if not full_df.empty:
            batch_df = full_df.sample(25, random_state=42).copy()
            st.info("Loaded 25 sample customer records from the Telco dataset.")

    if batch_df is not None:
        with st.spinner("Scoring batch dataset through ChurnLens model..."):
            scored_df = predict_batch(batch_df)

        st.markdown("### 📊 Batch Executive Summary")
        m1, m2, m3, m4 = st.columns(4)

        total_scored = len(scored_df)
        high_risk_count = (scored_df["Risk_Tier"] == "HIGH").sum()
        medium_risk_count = (scored_df["Risk_Tier"] == "MEDIUM").sum()
        low_risk_count = (scored_df["Risk_Tier"] == "LOW").sum()

        m1.metric("Total Customers Scored", f"{total_scored:,}")
        m2.metric(
            "High Risk Count",
            f"{high_risk_count}",
            f"{high_risk_count/total_scored*100:.1f}%",
            delta_color="inverse",
        )
        m3.metric(
            "Medium Risk Count",
            f"{medium_risk_count}",
            f"{medium_risk_count/total_scored*100:.1f}%",
            delta_color="off",
        )
        m4.metric(
            "Low Risk (Stable)",
            f"{low_risk_count}",
            f"{low_risk_count/total_scored*100:.1f}%",
        )

        st.markdown("### 📋 Scored Records")
        display_cols = [
            col
            for col in [
                "customerID",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "PaymentMethod",
                "Churn_Prediction",
                "Churn_Probability_%",
                "Risk_Tier",
            ]
            if col in scored_df.columns
        ]
        st.dataframe(scored_df[display_cols], use_container_width=True)

        csv_data = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Enriched Churn Risk CSV",
            data=csv_data,
            file_name="churnlens_scored_customers.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════════════════
# 3. EXECUTIVE BI DASHBOARD
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Executive BI Dashboard":
    st.subheader("📊 Executive Business Intelligence & Churn Trends")
    st.write(
        "Interactive analysis of retention drivers, contract exposures, and financial metrics across 7,032 Telco subscribers."
    )

    df = get_cleaned_data()
    if df.empty:
        st.warning("Cleaned dataset not found at data/cleaned/churn_cleaned.csv.")
    else:
        # High-level KPIs
        k1, k2, k3, k4 = st.columns(4)
        total_cust = len(df)
        churn_count = (df["Churn"] == 1).sum()
        churn_rate = churn_count / total_cust * 100
        churn_revenue = df[df["Churn"] == 1]["MonthlyCharges"].sum()

        k1.metric("Total Subscribers", f"{total_cust:,}")
        k2.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
        k3.metric("Churned Subscribers", f"{churn_count:,}")
        k4.metric("Monthly Revenue at Risk", f"${churn_revenue:,.2f}")

        st.markdown("---")

        c_v1, c_v2 = st.columns(2)

        with c_v1:
            st.markdown("#### 📄 Churn Rate by Contract Type")
            contract_stats = (
                df.groupby("Contract")
                .agg(Total=("Churn", "count"), Churned=("Churn", "sum"))
                .reset_index()
            )
            contract_stats["Churn_Rate_%"] = (contract_stats["Churned"] / contract_stats["Total"] * 100).round(1)

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                contract_stats["Contract"],
                contract_stats["Churn_Rate_%"],
                color=["#DC2626", "#F59E0B", "#10B981"],
            )
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, 50)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c_v2:
            st.markdown("#### 💳 Churn Rate by Payment Channel")
            pm_stats = (
                df.groupby("PaymentMethod")
                .agg(Total=("Churn", "count"), Churned=("Churn", "sum"))
                .reset_index()
            )
            pm_stats["Churn_Rate_%"] = (pm_stats["Churned"] / pm_stats["Total"] * 100).round(1)
            pm_stats = pm_stats.sort_values(by="Churn_Rate_%", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(pm_stats["PaymentMethod"], pm_stats["Churn_Rate_%"], color="#2563EB")
            ax.set_xlabel("Churn Rate (%)")
            for bar in bars:
                width = bar.get_width()
                ax.annotate(
                    f" {width}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    va="center",
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        c_v3, c_v4 = st.columns(2)

        with c_v3:
            st.markdown("#### ⏳ Customer Tenure vs Churn Propensity")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(
                data=df,
                x="tenure",
                hue="Churn",
                multiple="stack",
                bins=24,
                palette={0: "#3B82F6", 1: "#EF4444"},
                ax=ax,
            )
            ax.set_xlabel("Tenure in Months")
            ax.set_ylabel("Subscribers")
            ax.legend(["Churned", "Retained"])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c_v4:
            st.markdown("#### 💵 Monthly Charges Distribution by Churn")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                x="Churn",
                y="MonthlyCharges",
                data=df,
                palette={0: "#3B82F6", 1: "#EF4444"},
                ax=ax,
            )
            ax.set_xticklabels(["Retained ($61.31 avg)", "Churned ($74.44 avg)"])
            ax.set_ylabel("Monthly Charges ($)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ════════════════════════════════════════════════════════════════════
# 4. SQL ANALYTICS STUDIO
# ════════════════════════════════════════════════════════════════════
elif page == "🗄️ SQL Analytics Studio":
    st.subheader("🗄️ SQL Analytics Studio")
    st.write(
        "Execute production SQL queries against the Telco Customer dataset using the in-memory SQLite engine."
    )

    conn = get_db_connection()

    PRESET_QUERIES = {
        "1. Overall Churn Rate & Customer Volume": """SELECT 
    CASE WHEN Churn IN ('Yes', '1', 1) THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*) AS total_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
GROUP BY status;""",
        "2. Churn Rate by Contract Type": """SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;""",
        "3. Monthly Charges & Revenue Exposure": """SELECT 
    CASE WHEN Churn IN ('Yes', '1', 1) THEN 'Churned' ELSE 'Retained' END AS customer_status,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(AVG(TotalCharges), 2) AS avg_lifetime_value
FROM customers
GROUP BY customer_status;""",
        "4. Churn Distribution by Tenure Cohort": """SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-1 Year (Onboarding)'
        WHEN tenure <= 24 THEN '1-2 Years (Early Stage)'
        WHEN tenure <= 48 THEN '2-4 Years (Established)'
        ELSE '4+ Years (Loyal)'
    END AS tenure_cohort,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_cohort
ORDER BY MIN(tenure) ASC;""",
        "5. Churn Rate by Payment Channel": """SELECT 
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;""",
        "6. High-Value Customer Risk (Charges >= $70)": """SELECT 
    Contract,
    COUNT(*) AS high_value_customers,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM customers
WHERE MonthlyCharges >= 70.0
GROUP BY Contract
ORDER BY monthly_revenue_at_risk DESC;""",
        "7. Bundled Tech Support & Cyber Protection Impact": """SELECT 
    InternetService,
    TechSupport,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE InternetService != 'No'
GROUP BY InternetService, TechSupport
ORDER BY InternetService, churn_rate_pct DESC;""",
    }

    selected_query_title = st.selectbox("Choose a Pre-Built Business Intelligence Query:", list(PRESET_QUERIES.keys()))
    default_sql = PRESET_QUERIES[selected_query_title]

    custom_sql = st.text_area("SQL Editor (Table name: customers)", value=default_sql, height=160)

    if st.button("▶️ Execute Query", use_container_width=True):
        try:
            query_res = pd.read_sql_query(custom_sql, conn)
            st.success(f"Query returned {len(query_res)} rows.")
            st.dataframe(query_res, use_container_width=True)

            csv_sql = query_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Query Results (CSV)",
                data=csv_sql,
                file_name="churnlens_query_result.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"SQL Error: {e}")


# ════════════════════════════════════════════════════════════════════
# 5. MODEL GOVERNANCE & CARD
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Governance & Card":
    st.subheader("🤖 Model Governance & Architecture")
    st.write(
        "Transparency, performance benchmarks, and feature attribution details for the production ChurnLens classifier."
    )

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Model Architecture", "Random Forest")
    g2.metric("Accuracy", "78.89%")
    g3.metric("ROC-AUC Score", "0.8248")
    g4.metric("Recall (Churn)", "50.27%")
    g5.metric("Precision (Churn)", "62.88%")

    st.markdown("---")
    c_art1, c_art2 = st.columns(2)

    with c_art1:
        st.markdown("#### 🎯 Confusion Matrix")
        cm_path = VISUALS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Random Forest Confusion Matrix on Stratified Test Holdout")
        else:
            st.info("Run `python scripts/train.py` to generate the latest confusion matrix.")

    with c_art2:
        st.markdown("#### 🏆 Top Predictive Features")
        feat_path = VISUALS_DIR / "feature_importance.png"
        if feat_path.exists():
            st.image(str(feat_path), caption="Gini Feature Importance Rankings")
        else:
            st.info("Run `python scripts/train.py` to generate feature importances.")

    st.markdown("### 📋 Model Card Specification")
    st.markdown(
        """
        | Attribute | Specification |
        | :--- | :--- |
        | **Model Class** | `sklearn.ensemble.RandomForestClassifier` |
        | **Hyperparameters** | `n_estimators=100, max_depth=12, min_samples_split=5, random_state=42` |
        | **Dataset** | Telco Customer Churn (7,032 sanitized records post-null filtering) |
        | **Validation Strategy** | 80/20 Stratified Train/Test Split |
        | **Primary Evaluation Metric** | ROC-AUC (0.8248) and F1-Score (0.5587) |
        | **Target Encoding** | `0 = Retained, 1 = Churned` |
        | **Model Artifact Size** | Compressed joblib (~1.73 MB) |
        """
    )


# ════════════════════════════════════════════════════════════════════
# 6. ABOUT & ATTRIBUTION
# ════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About & Attribution":
    st.subheader("ℹ️ About ChurnLens")
    st.markdown(
        """
        ### 🌐 Project Overview
        **ChurnLens** is an enterprise-grade Customer Churn Prediction and Business Intelligence suite
        designed and developed by **Aanjaneya Pandey**.
        It combines predictive machine learning, SQL analytics, and actionable retention playbooks to help
        businesses protect recurring revenue and proactively engage at-risk accounts.

        ---

        ### 👤 Author & Maintainer
        - **Author**: **Aanjaneya Pandey**
        - **GitHub**: [Aanjaneya24](https://github.com/Aanjaneya24)
        - **Repository**: [https://github.com/Aanjaneya24/ChurnLens](https://github.com/Aanjaneya24/ChurnLens)
        - **Live App**: [https://churnlens-usz4.onrender.com](https://churnlens-usz4.onrender.com)
        - **Tech Stack**: Python, Scikit-learn, Pandas, Streamlit, SQL, Joblib, Matplotlib, Seaborn

        ---

        ### ⚖️ License
        This project is licensed under the **MIT License**. Copyright (c) 2026 Aanjaneya Pandey.
        """
    )