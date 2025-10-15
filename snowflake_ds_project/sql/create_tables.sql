-- ============================================================================
-- Snowflake Database Setup for E-Commerce Churn Prediction
-- ============================================================================

-- Create database and schemas
CREATE DATABASE IF NOT EXISTS ECOMMERCE_DB;

USE DATABASE ECOMMERCE_DB;

CREATE SCHEMA IF NOT EXISTS RAW_DATA;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;
CREATE SCHEMA IF NOT EXISTS FEATURES;
CREATE SCHEMA IF NOT EXISTS ML_MODELS;

-- ============================================================================
-- RAW_DATA Schema: Landing zone for raw data
-- ============================================================================

USE SCHEMA RAW_DATA;

-- Customers table
CREATE OR REPLACE TABLE CUSTOMERS (
    customer_id VARCHAR(50) PRIMARY KEY,
    registration_date DATE,
    age INTEGER,
    gender VARCHAR(20),
    country VARCHAR(100),
    membership_tier VARCHAR(20),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Transactions table
CREATE OR REPLACE TABLE TRANSACTIONS (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    transaction_date TIMESTAMP_NTZ,
    amount DECIMAL(10, 2),
    product_category VARCHAR(100),
    quantity INTEGER,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (customer_id) REFERENCES CUSTOMERS(customer_id)
);

-- Customer activity table
CREATE OR REPLACE TABLE CUSTOMER_ACTIVITY (
    activity_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    last_login_date DATE,
    page_views INTEGER,
    support_tickets INTEGER,
    email_opened INTEGER,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (customer_id) REFERENCES CUSTOMERS(customer_id)
);

-- ============================================================================
-- ANALYTICS Schema: Cleaned and transformed data
-- ============================================================================

USE SCHEMA ANALYTICS;

-- Consolidated customer view
CREATE OR REPLACE VIEW CUSTOMER_360 AS
SELECT 
    c.customer_id,
    c.registration_date,
    c.age,
    c.gender,
    c.country,
    c.membership_tier,
    ca.last_login_date,
    ca.page_views,
    ca.support_tickets,
    ca.email_opened,
    COUNT(DISTINCT t.transaction_id) as total_transactions,
    SUM(t.amount) as total_spend,
    AVG(t.amount) as avg_transaction_value,
    MAX(t.transaction_date) as last_transaction_date,
    DATEDIFF(day, MAX(t.transaction_date), CURRENT_DATE()) as days_since_last_purchase
FROM ECOMMERCE_DB.RAW_DATA.CUSTOMERS c
LEFT JOIN ECOMMERCE_DB.RAW_DATA.CUSTOMER_ACTIVITY ca 
    ON c.customer_id = ca.customer_id
LEFT JOIN ECOMMERCE_DB.RAW_DATA.TRANSACTIONS t 
    ON c.customer_id = t.customer_id
GROUP BY 
    c.customer_id, c.registration_date, c.age, c.gender, 
    c.country, c.membership_tier, ca.last_login_date, 
    ca.page_views, ca.support_tickets, ca.email_opened;

-- ============================================================================
-- FEATURES Schema: Feature engineering tables
-- ============================================================================

USE SCHEMA FEATURES;

-- Customer features for ML model
CREATE OR REPLACE TABLE CUSTOMER_FEATURES AS
SELECT 
    customer_id,
    age,
    CASE 
        WHEN gender = 'Male' THEN 1
        WHEN gender = 'Female' THEN 0
        ELSE -1
    END as gender_encoded,
    CASE 
        WHEN membership_tier = 'Gold' THEN 3
        WHEN membership_tier = 'Silver' THEN 2
        WHEN membership_tier = 'Bronze' THEN 1
        ELSE 0
    END as membership_tier_encoded,
    DATEDIFF(day, registration_date, CURRENT_DATE()) as days_since_registration,
    COALESCE(total_transactions, 0) as total_transactions,
    COALESCE(total_spend, 0) as total_spend,
    COALESCE(avg_transaction_value, 0) as avg_transaction_value,
    COALESCE(days_since_last_purchase, 999) as days_since_last_purchase,
    COALESCE(page_views, 0) as page_views,
    COALESCE(support_tickets, 0) as support_tickets,
    COALESCE(email_opened, 0) as email_opened,
    CASE 
        WHEN COALESCE(days_since_last_purchase, 999) > 90 THEN 1
        ELSE 0
    END as is_churned
FROM ECOMMERCE_DB.ANALYTICS.CUSTOMER_360;

-- ============================================================================
-- ML_MODELS Schema: Model metadata and predictions
-- ============================================================================

USE SCHEMA ML_MODELS;

-- Model metadata table
CREATE OR REPLACE TABLE MODEL_METADATA (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100),
    model_type VARCHAR(50),
    training_date TIMESTAMP_NTZ,
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    auc_roc DECIMAL(5, 4),
    feature_importance VARIANT,
    hyperparameters VARIANT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Predictions table
CREATE OR REPLACE TABLE PREDICTIONS (
    prediction_id VARCHAR(50) PRIMARY KEY,
    model_id VARCHAR(50),
    customer_id VARCHAR(50),
    prediction_date TIMESTAMP_NTZ,
    churn_probability DECIMAL(5, 4),
    predicted_churn INTEGER,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (model_id) REFERENCES MODEL_METADATA(model_id)
);

-- ============================================================================
-- Create warehouse for compute
-- ============================================================================

CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH
    WITH WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- Grant permissions (adjust as needed for your security requirements)
GRANT USAGE ON DATABASE ECOMMERCE_DB TO ROLE ACCOUNTADMIN;
GRANT USAGE ON ALL SCHEMAS IN DATABASE ECOMMERCE_DB TO ROLE ACCOUNTADMIN;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN DATABASE ECOMMERCE_DB TO ROLE ACCOUNTADMIN;
GRANT SELECT ON ALL VIEWS IN DATABASE ECOMMERCE_DB TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE ACCOUNTADMIN;

