# Complete Data Science Project Guide: Customer Churn Prediction with Snowflake

**Author:** Manus AI  
**Date:** October 14, 2025

## Executive Summary

This document provides a comprehensive guide to a complete, production-ready data science project that demonstrates how to leverage **Snowflake** as a central data warehouse for machine learning workflows. The project focuses on predicting customer churn for an e-commerce platform using historical transaction data, customer demographics, and behavioral patterns.

The project showcases industry best practices for data engineering, feature engineering, model training, and prediction storage, all orchestrated through Python and Snowflake. It is designed to be fully reproducible, modular, and scalable, making it an excellent template for real-world data science applications.

## Project Architecture

The architecture follows a modern data science workflow that separates concerns across multiple layers, each with a specific responsibility. The diagram below illustrates the high-level architecture:

### Architecture Layers

**1. Data Layer (Snowflake)**

The data layer consists of four schemas within the `ECOMMERCE_DB` database, each serving a distinct purpose in the data lifecycle:

| Schema | Purpose | Key Tables/Views |
|--------|---------|------------------|
| `RAW_DATA` | Landing zone for raw, unprocessed data | `CUSTOMERS`, `TRANSACTIONS`, `CUSTOMER_ACTIVITY` |
| `ANALYTICS` | Transformed data and analytical views | `CUSTOMER_360` (view) |
| `FEATURES` | Feature-engineered tables for ML | `CUSTOMER_FEATURES` |
| `ML_MODELS` | Model metadata and predictions | `MODEL_METADATA`, `PREDICTIONS` |

This layered approach follows the **medallion architecture** pattern (Bronze → Silver → Gold), which is a best practice in modern data engineering. It provides clear separation between raw data, transformed data, and analytics-ready datasets.

**2. Application Layer (Python)**

The application layer is organized into modular Python components, each responsible for a specific aspect of the workflow:

- **Configuration Module** (`config/`): Manages Snowflake connection parameters and environment variables.
- **Data Generation** (`data/`): Creates synthetic e-commerce data for reproducibility.
- **Core Modules** (`src/`):
  - `snowflake_connector.py`: Wrapper for Snowflake database operations.
  - `data_loader.py`: Handles data ingestion from CSV to Snowflake.
  - `feature_engineering.py`: Extracts and prepares features from Snowflake.
  - `model_training.py`: Trains, evaluates, and saves ML models.
- **Interactive Notebooks** (`notebooks/`): Step-by-step Jupyter notebooks for exploration and execution.

**3. Workflow Orchestration**

The project follows a sequential workflow:

1. **Data Generation** → Generate synthetic e-commerce data
2. **Data Ingestion** → Load raw data into Snowflake
3. **Feature Engineering** → Transform data and create ML features
4. **Exploratory Analysis** → Analyze patterns and insights
5. **Model Training** → Train and evaluate ML models
6. **Prediction & Storage** → Generate predictions and store in Snowflake

## Data Schema and Features

### Raw Data Tables

The project uses three primary raw data tables:

**CUSTOMERS Table**

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | VARCHAR(50) | Unique customer identifier |
| `registration_date` | DATE | Date when customer registered |
| `age` | INTEGER | Customer age |
| `gender` | VARCHAR(20) | Customer gender |
| `country` | VARCHAR(100) | Customer country |
| `membership_tier` | VARCHAR(20) | Membership level (Bronze/Silver/Gold) |

**TRANSACTIONS Table**

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | VARCHAR(50) | Unique transaction identifier |
| `customer_id` | VARCHAR(50) | Foreign key to CUSTOMERS |
| `transaction_date` | TIMESTAMP_NTZ | Transaction timestamp |
| `amount` | DECIMAL(10, 2) | Transaction amount |
| `product_category` | VARCHAR(100) | Product category |
| `quantity` | INTEGER | Number of items purchased |

**CUSTOMER_ACTIVITY Table**

| Column | Type | Description |
|--------|------|-------------|
| `activity_id` | VARCHAR(50) | Unique activity record identifier |
| `customer_id` | VARCHAR(50) | Foreign key to CUSTOMERS |
| `last_login_date` | DATE | Most recent login date |
| `page_views` | INTEGER | Total page views |
| `support_tickets` | INTEGER | Number of support tickets |
| `email_opened` | INTEGER | Number of emails opened |

### Feature Engineering

The feature engineering process transforms raw data into machine learning features through SQL-based transformations in Snowflake. The `CUSTOMER_FEATURES` table contains the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Customer age | Numeric |
| `gender_encoded` | Gender (1=Male, 0=Female, -1=Other) | Categorical |
| `membership_tier_encoded` | Tier (3=Gold, 2=Silver, 1=Bronze) | Categorical |
| `days_since_registration` | Days since customer registered | Numeric |
| `total_transactions` | Total number of transactions | Numeric |
| `total_spend` | Total amount spent | Numeric |
| `avg_transaction_value` | Average transaction amount | Numeric |
| `days_since_last_purchase` | Days since last transaction | Numeric |
| `page_views` | Total page views | Numeric |
| `support_tickets` | Number of support tickets | Numeric |
| `email_opened` | Number of emails opened | Numeric |
| `is_churned` | Target variable (1=churned, 0=active) | Binary |

The target variable `is_churned` is defined as customers who have not made a purchase in the last 90 days. This business rule can be adjusted based on specific requirements.

## Machine Learning Pipeline

### Model Training Process

The machine learning pipeline implements a comprehensive approach to churn prediction:

**1. Data Preparation**

The data preparation phase includes several critical steps to ensure model quality:

- **Feature Extraction**: Features are loaded directly from Snowflake using SQL queries, leveraging Snowflake's computational power.
- **Train-Test Split**: Data is split into 80% training and 20% testing sets, with stratification to maintain class distribution.
- **Feature Scaling**: Numerical features are standardized using `StandardScaler` to ensure all features contribute equally to the model.
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the classes, addressing the typical imbalance in churn datasets where churned customers are the minority.

**2. Model Selection and Training**

The project trains and compares four different classification algorithms:

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| **Logistic Regression** | Linear model for binary classification | C=1.0, max_iter=1000 |
| **Random Forest** | Ensemble of decision trees | n_estimators=100, random_state=42 |
| **Gradient Boosting** | Sequential ensemble method | n_estimators=100, learning_rate=0.1 |
| **XGBoost** | Optimized gradient boosting | n_estimators=100, max_depth=5, learning_rate=0.1 |

Each model is trained on the balanced training set and evaluated on the original (unbalanced) test set to reflect real-world performance.

**3. Model Evaluation**

Models are evaluated using multiple metrics to provide a comprehensive view of performance:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted churners who actually churned
- **Recall**: Proportion of actual churners correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

The XGBoost model typically achieves the best performance across these metrics, making it the default choice for production deployment.

**4. Feature Importance Analysis**

After training, the project analyzes which features contribute most to the predictions. This provides valuable business insights:

- **High-importance features** indicate key drivers of churn
- **Low-importance features** may be candidates for removal in future iterations
- Feature importance helps validate that the model is learning meaningful patterns

**5. Prediction and Storage**

Once the best model is selected, the pipeline:

1. Generates churn probability scores for all customers
2. Applies a decision threshold (default: 0.5) to classify customers as churned or active
3. Saves model metadata to the `MODEL_METADATA` table in Snowflake
4. Saves individual predictions to the `PREDICTIONS` table in Snowflake

This approach ensures that predictions are immediately available for downstream applications, such as marketing automation or customer success tools.

## Implementation Guide

### Prerequisites

Before starting, ensure you have:

1. **Snowflake Account**: A Snowflake account with appropriate permissions (ACCOUNTADMIN role recommended for this demo)
2. **Python 3.11**: The project is built and tested with Python 3.11
3. **Virtual Environment**: Recommended for dependency isolation
4. **Git**: For cloning the repository (if applicable)

### Step-by-Step Setup

**Step 1: Environment Setup**

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Configure Snowflake Connection**

Create a `.env` file in the project root with your Snowflake credentials:

```
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=ECOMMERCE_DB
SNOWFLAKE_SCHEMA=ANALYTICS
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

**Step 3: Generate Synthetic Data**

```bash
python data/generate_synthetic_data.py
```

This creates three CSV files in the `data/` directory with 10,000 customers and approximately 150,000 transactions.

**Step 4: Execute Notebooks**

Launch Jupyter and run the notebooks in sequence:

```bash
jupyter lab
```

Execute in this order:
1. `01_data_ingestion.ipynb` - Creates Snowflake schema and loads data
2. `02_eda.ipynb` - Performs exploratory data analysis
3. `03_feature_engineering.ipynb` - Creates feature tables
4. `04_modeling.ipynb` - Trains models and generates predictions

### Alternative: Script-Based Execution

For automated execution without notebooks:

```bash
# Generate data
python data/generate_synthetic_data.py

# Load data to Snowflake
python src/data_loader.py

# Train model and generate predictions
python src/model_training.py
```

## Key Insights and Best Practices

### Snowflake Best Practices

**1. Schema Organization**

The project demonstrates proper schema organization by separating raw data, analytics, features, and model outputs. This separation provides several benefits:

- **Clear data lineage**: Easy to trace data from source to prediction
- **Access control**: Different teams can have different permissions on each schema
- **Performance optimization**: Queries are faster when they target specific schemas

**2. Compute Warehouse Management**

The project uses a single `COMPUTE_WH` warehouse with auto-suspend and auto-resume enabled. In production, consider:

- **Separate warehouses** for different workloads (ETL, analytics, ML)
- **Warehouse sizing** based on query complexity and data volume
- **Query optimization** using clustering keys and materialized views

**3. Data Loading**

The project uses the `snowflake-connector-python` library with the `write_pandas` function for efficient data loading. For larger datasets, consider:

- **Snowflake Stages**: Use internal or external stages for bulk loading
- **COPY INTO**: More efficient for very large files
- **Snowpipe**: For continuous, automated data ingestion

### Machine Learning Best Practices

**1. Feature Engineering in SQL**

By performing feature engineering in Snowflake using SQL, the project leverages:

- **Scalability**: Snowflake can handle large datasets efficiently
- **Consistency**: Same features used in training and production
- **Maintainability**: SQL is widely understood and easy to modify

**2. Model Versioning**

The `MODEL_METADATA` table stores key information about each trained model, including:

- Training date and model type
- Performance metrics
- Feature importance
- Hyperparameters

This enables model versioning and comparison over time.

**3. Prediction Storage**

Storing predictions in Snowflake provides several advantages:

- **Accessibility**: Predictions are immediately available to downstream systems
- **Auditability**: Complete history of predictions for compliance and debugging
- **Integration**: Easy to join predictions with other business data

## Extending the Project

This project serves as a foundation that can be extended in multiple ways:

### Production Deployment

**1. Automated Pipeline**

Convert the notebooks into production scripts and orchestrate with tools like:

- **Apache Airflow**: For complex DAG-based workflows
- **Prefect**: For modern, Python-native orchestration
- **dbt**: For SQL-based transformations

**2. Model Serving**

Deploy the trained model as a REST API using:

- **Flask/FastAPI**: For lightweight API development
- **Docker**: For containerization
- **Kubernetes**: For scalable deployment

**3. Monitoring and Alerting**

Implement monitoring for:

- **Data quality**: Detect anomalies in incoming data
- **Model performance**: Track metrics over time
- **Prediction drift**: Identify when model needs retraining

### Advanced Features

**1. Hyperparameter Tuning**

The project includes a `tune_hyperparameters` parameter in the training function. Enable this for:

- Grid search or random search over hyperparameter space
- Cross-validation for robust performance estimation
- Automated model selection

**2. Additional Features**

Consider engineering additional features:

- **Temporal features**: Day of week, month, seasonality
- **Interaction features**: Combinations of existing features
- **External data**: Economic indicators, weather, events

**3. Real-Time Scoring**

Implement real-time churn prediction:

- **Snowflake Streams**: Capture changes in customer data
- **Snowflake Tasks**: Trigger predictions on new data
- **External Functions**: Call Python model from Snowflake

## Conclusion

This project demonstrates a complete, production-quality data science workflow using Snowflake as the central data platform. It showcases best practices in data engineering, feature engineering, model training, and prediction storage, all while maintaining code quality, modularity, and reproducibility.

The modular design makes it easy to adapt this project to other use cases, such as fraud detection, customer lifetime value prediction, or product recommendation. By following the patterns established in this project, data scientists and engineers can build robust, scalable machine learning systems that deliver real business value.

## Additional Resources

For further learning and exploration:

- **Snowflake Documentation**: [https://docs.snowflake.com](https://docs.snowflake.com)
- **scikit-learn User Guide**: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- **XGBoost Documentation**: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- **Snowflake for Data Science**: [https://www.snowflake.com/workloads/data-science/](https://www.snowflake.com/workloads/data-science/)

---

**Project Repository**: This project is fully open-source and available for modification and extension. Feel free to adapt it to your specific use case and requirements.

