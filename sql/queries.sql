-- ==============================================================================
-- ChurnLens — Customer Churn Business Intelligence & SQL Analytics
-- Dataset: Telco Customer Churn
-- Table Name: customers
-- Compatible with: SQLite, MySQL, PostgreSQL, DuckDB
-- ==============================================================================

-- 1. Overall Churn Rate & Customer Volume
-- Measures the primary baseline retention and churn metric across the customer base.
SELECT 
    CASE 
        WHEN Churn IN ('Yes', '1', 1) THEN 'Churned'
        ELSE 'Retained'
    END AS status,
    COUNT(*) AS total_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
GROUP BY status;

-- 2. Churn Rate by Contract Type
-- Evaluates the impact of contract duration (Month-to-month vs One year vs Two year) on churn.
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Monthly Charges & Revenue Impact by Churn Status
-- Quantifies the financial exposure and average billing of churned vs retained customers.
SELECT 
    CASE 
        WHEN Churn IN ('Yes', '1', 1) THEN 'Churned'
        ELSE 'Retained'
    END AS customer_status,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(AVG(TotalCharges), 2) AS avg_lifetime_value
FROM customers
GROUP BY customer_status;

-- 4. Churn Distribution by Tenure Cohort
-- Tracks customer lifecycle stages to identify when customers are most vulnerable to churn.
SELECT 
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
ORDER BY MIN(tenure) ASC;

-- 5. Churn Rate by Payment Method
-- Identifies friction points and churn risk correlated with specific payment channels.
SELECT 
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 6. High-Value Customer Risk Analysis (Monthly Charges > $70)
-- Isolates premium accounts to analyze where the highest revenue losses occur.
SELECT 
    Contract,
    COUNT(*) AS high_value_customers,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk
FROM customers
WHERE MonthlyCharges >= 70.0
GROUP BY Contract
ORDER BY monthly_revenue_at_risk DESC;

-- 7. Service Bundling & Tech Support Impact on Retention
-- Measures how value-added services (Tech Support & Online Security) reduce churn probability.
SELECT 
    InternetService,
    TechSupport,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN Churn IN ('Yes', '1', 1) THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE InternetService != 'No'
GROUP BY InternetService, TechSupport
ORDER BY InternetService, churn_rate_pct DESC;