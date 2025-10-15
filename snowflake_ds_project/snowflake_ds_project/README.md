# End-to-End Data Science Project with Snowflake

This repository contains a complete, end-to-end data science project demonstrating how to use Snowflake for data warehousing, analysis, and machine learning. The project focuses on predicting customer churn for an e-commerce platform.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Execution Guide](#execution-guide)
- [Data Schema](#data-schema)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to build a machine learning model that predicts which customers are likely to churn (i.e., stop making purchases). By identifying at-risk customers, the e-commerce business can proactively engage them with targeted marketing campaigns, special offers, or support outreach to retain them.

The project leverages Snowflake as the central data warehouse for storing raw data, transformed analytical tables, engineered features, and model predictions. The entire workflow, from data ingestion to model training and prediction, is orchestrated through Python scripts and Jupyter notebooks that interact with Snowflake.

## Features

- **End-to-End Workflow:** Covers the entire data science lifecycle from data generation to model deployment.
- **Snowflake Integration:** Demonstrates best practices for using Snowflake in a data science context, including:
  - Role-based access control and warehouse management.
  - Loading data using the `snowflake-connector-python`.
  - Performing transformations and feature engineering using SQL.
  - Storing model metadata and predictions back into Snowflake.
- **Synthetic Data Generation:** Includes a script to generate realistic e-commerce data, making the project fully reproducible without needing external datasets.
- **Modular Code:** The project is organized into a clean, modular structure with separate components for configuration, data loading, feature engineering, and model training.
- **Jupyter Notebooks:** Provides a step-by-step, interactive guide through the data analysis and modeling process.

## Technology Stack

- **Data Warehouse:** Snowflake
- **Programming Language:** Python 3.11
- **Core Python Libraries:**
  - `snowflake-connector-python`: For connecting to and interacting with Snowflake.
  - `pandas` & `numpy`: For data manipulation.
  - `scikit-learn` & `xgboost`: For machine learning.
  - `matplotlib`, `seaborn`, `plotly`: For data visualization.
  - `jupyter`: For interactive analysis and notebooks.

## Project Structure

```
snowflake_ds_project/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── __init__.py
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
    └── create_tables.sql
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd snowflake_ds_project
```

### 2. Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Snowflake Connection

Create a `.env` file in the project root directory by copying the example file.

```bash
cp .env.example .env
```

Now, edit the `.env` file and fill in your Snowflake account details:

```
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=ECOMMERCE_DB
SNOWFLAKE_SCHEMA=ANALYTICS
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

**Note:** The `ACCOUNT_IDENTIFIER` can be found in the URL you use to access Snowflake (e.g., `my-account.snowflakecomputing.com`).

## Execution Guide

Follow these steps to run the project from start to finish.

### Step 1: Generate Synthetic Data

Run the script to generate the raw CSV data files for customers, transactions, and activities. The files will be saved in the `data/` directory.

```bash
python data/generate_synthetic_data.py
```

### Step 2: Set Up Snowflake Database and Tables

Before loading the data, you need to create the necessary database, schemas, and tables in Snowflake. You can do this by executing the `create_tables.sql` script in your Snowflake worksheet.

Alternatively, the data loading notebook (`01_data_ingestion.ipynb`) also includes a step to execute this script automatically.

### Step 3: Run the Jupyter Notebooks

Launch Jupyter Lab or Jupyter Notebook to run the notebooks in the `notebooks/` directory.

```bash
jupyter lab
```

Run the notebooks in the following order:

1.  **`01_data_ingestion.ipynb`**: This notebook loads the generated CSV data into the `RAW_DATA` schema in Snowflake.
2.  **`02_eda.ipynb`**: Performs exploratory data analysis on the data in Snowflake to uncover insights about customer demographics, transaction patterns, and churn behavior.
3.  **`03_feature_engineering.ipynb`**: This notebook runs the SQL queries to create the `CUSTOMER_360` view and the final `CUSTOMER_FEATURES` table for modeling.
4.  **`04_modeling.ipynb`**: This is the core machine learning notebook. It trains several classification models, evaluates their performance, selects the best model (XGBoost), and saves the model metadata and predictions back to Snowflake.

## Data Schema

The project uses the following schemas in the `ECOMMERCE_DB` database:

-   **`RAW_DATA`**: Contains the raw, unprocessed data loaded from the CSV files (`CUSTOMERS`, `TRANSACTIONS`, `CUSTOMER_ACTIVITY`).
-   **`ANALYTICS`**: Contains transformed data and views for analysis, including the `CUSTOMER_360` view which provides a consolidated view of each customer.
-   **`FEATURES`**: Contains the final feature table (`CUSTOMER_FEATURES`) used for training the machine learning model.
-   **`ML_MODELS`**: Stores metadata about the trained models (`MODEL_METADATA`) and the churn predictions for each customer (`PREDICTIONS`).

## Machine Learning Pipeline

The machine learning pipeline consists of the following steps, primarily executed in the `04_modeling.ipynb` notebook:

1.  **Data Preparation**: Features are loaded from the `FEATURES.CUSTOMER_FEATURES` table in Snowflake.
2.  **Train-Test Split**: The data is split into training and testing sets.
3.  **Data Scaling**: Numerical features are standardized using `StandardScaler`.
4.  **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to address the class imbalance between churned and active customers.
5.  **Model Training**: Several models are trained, including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
6.  **Model Evaluation**: Models are evaluated based on Accuracy, Precision, Recall, F1-Score, and ROC AUC.
7.  **Model Selection**: The XGBoost model is selected as the best-performing model.
8.  **Feature Importance**: The feature importances from the XGBoost model are analyzed to understand the key drivers of churn.
9.  **Saving Results**: The best model's metadata and the churn predictions for all customers are saved back to the `ML_MODELS` schema in Snowflake.

## Conclusion

This project provides a comprehensive template for building and operationalizing a data science project using Snowflake. It demonstrates a robust workflow that is both scalable and maintainable, making it an excellent starting point for real-world machine learning applications.

