# Snowflake Data Science Project Design

## Project Overview
**Title:** E-Commerce Customer Churn Prediction with Snowflake

**Use Case:** Predict customer churn for an e-commerce platform using historical transaction data, customer demographics, and behavioral patterns stored in Snowflake.

## Architecture Components

### 1. Data Layer (Snowflake)
- **Database:** ECOMMERCE_DB
- **Schemas:**
  - RAW_DATA: Landing zone for raw data
  - ANALYTICS: Cleaned and transformed data
  - FEATURES: Feature engineering tables
  - ML_MODELS: Model metadata and predictions

### 2. Data Pipeline
- Synthetic data generation (simulating real e-commerce data)
- Data ingestion to Snowflake using Snowflake Connector for Python
- Data transformation using Snowflake SQL and stored procedures
- Feature engineering with Snowflake's analytical functions

### 3. Analytics & ML Components
- Exploratory Data Analysis (EDA) using data from Snowflake
- Feature engineering leveraging Snowflake's compute
- Machine Learning model training (scikit-learn + XGBoost)
- Model evaluation and validation
- Predictions stored back to Snowflake

### 4. Technology Stack
- **Data Warehouse:** Snowflake
- **Programming:** Python 3.11
- **Libraries:**
  - snowflake-connector-python
  - snowflake-snowpark-python
  - pandas, numpy
  - scikit-learn, xgboost
  - matplotlib, seaborn, plotly
- **Environment:** Jupyter notebooks + Python scripts

## Data Schema

### Customer Table
- customer_id, registration_date, age, gender, country, membership_tier

### Transactions Table
- transaction_id, customer_id, transaction_date, amount, product_category, quantity

### Customer Activity Table
- customer_id, last_login_date, page_views, support_tickets, email_opened

### Target Variable
- Churn (binary): 1 if customer hasn't made purchase in last 90 days, 0 otherwise

## Project Structure
```
snowflake_ds_project/
├── README.md
├── requirements.txt
├── config/
│   └── snowflake_config.py
├── data/
│   └── generate_synthetic_data.py
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── snowflake_connector.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── model_training.py
└── sql/
    ├── create_tables.sql
    ├── transform_data.sql
    └── create_features.sql
```

## Key Features
1. End-to-end ML pipeline with Snowflake integration
2. Demonstrates Snowflake best practices (stages, warehouses, role-based access)
3. Feature engineering using Snowflake SQL
4. Model training with data pulled from Snowflake
5. Prediction results stored back to Snowflake
6. Comprehensive documentation and reproducible code

