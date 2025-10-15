"""
Feature Engineering Module

This module handles feature extraction and engineering from Snowflake data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging
from .snowflake_connector import SnowflakeConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Class for feature engineering operations.
    """
    
    def __init__(self, connector: SnowflakeConnector):
        """
        Initialize the FeatureEngineer.
        
        Args:
            connector: SnowflakeConnector instance
        """
        self.connector = connector
    
    def get_features_from_snowflake(self) -> pd.DataFrame:
        """
        Retrieve engineered features from Snowflake.
        
        Returns:
            DataFrame with features
        """
        query = """
        SELECT 
            customer_id,
            age,
            gender_encoded,
            membership_tier_encoded,
            days_since_registration,
            total_transactions,
            total_spend,
            avg_transaction_value,
            days_since_last_purchase,
            page_views,
            support_tickets,
            email_opened,
            is_churned
        FROM ECOMMERCE_DB.FEATURES.CUSTOMER_FEATURES
        """
        
        logger.info("Fetching features from Snowflake...")
        df = self.connector.execute_query_to_df(query)
        logger.info(f"Retrieved {len(df)} feature records")
        
        return df
    
    def prepare_ml_dataset(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare dataset for machine learning.
        
        Returns:
            Tuple of (features DataFrame, target Series, feature names list)
        """
        # Get features from Snowflake
        df = self.get_features_from_snowflake()
        
        # Separate features and target
        feature_columns = [
            'age',
            'gender_encoded',
            'membership_tier_encoded',
            'days_since_registration',
            'total_transactions',
            'total_spend',
            'avg_transaction_value',
            'days_since_last_purchase',
            'page_views',
            'support_tickets',
            'email_opened'
        ]
        
        X = df[feature_columns]
        y = df['is_churned']
        
        logger.info(f"Prepared ML dataset with {len(X)} samples and {len(feature_columns)} features")
        logger.info(f"Churn rate: {y.mean():.2%}")
        
        return X, y, feature_columns
    
    def get_customer_360_view(self, limit: int = 1000) -> pd.DataFrame:
        """
        Get comprehensive customer view for analysis.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with customer 360 view
        """
        query = f"""
        SELECT *
        FROM ECOMMERCE_DB.ANALYTICS.CUSTOMER_360
        LIMIT {limit}
        """
        
        logger.info(f"Fetching customer 360 view (limit: {limit})...")
        df = self.connector.execute_query_to_df(query)
        logger.info(f"Retrieved {len(df)} customer records")
        
        return df
    
    def get_transaction_summary(self) -> pd.DataFrame:
        """
        Get transaction summary statistics.
        
        Returns:
            DataFrame with transaction summaries
        """
        query = """
        SELECT 
            product_category,
            COUNT(*) as transaction_count,
            SUM(amount) as total_revenue,
            AVG(amount) as avg_transaction_value,
            SUM(quantity) as total_quantity
        FROM ECOMMERCE_DB.RAW_DATA.TRANSACTIONS
        GROUP BY product_category
        ORDER BY total_revenue DESC
        """
        
        logger.info("Fetching transaction summary...")
        df = self.connector.execute_query_to_df(query)
        
        return df
    
    def get_churn_statistics(self) -> pd.DataFrame:
        """
        Get churn statistics by various dimensions.
        
        Returns:
            DataFrame with churn statistics
        """
        query = """
        SELECT 
            membership_tier,
            country,
            COUNT(*) as total_customers,
            SUM(is_churned) as churned_customers,
            AVG(is_churned) as churn_rate,
            AVG(total_spend) as avg_spend,
            AVG(total_transactions) as avg_transactions
        FROM ECOMMERCE_DB.FEATURES.CUSTOMER_FEATURES cf
        JOIN ECOMMERCE_DB.RAW_DATA.CUSTOMERS c 
            ON cf.customer_id = c.customer_id
        GROUP BY membership_tier, country
        ORDER BY churn_rate DESC
        """
        
        logger.info("Fetching churn statistics...")
        df = self.connector.execute_query_to_df(query)
        
        return df


def main():
    """Main function to demonstrate feature engineering."""
    from config import config
    
    # Validate configuration
    config.validate()
    
    # Connect to Snowflake
    with SnowflakeConnector(config.get_connection_params()) as connector:
        # Create feature engineer
        engineer = FeatureEngineer(connector)
        
        # Get features
        X, y, feature_names = engineer.prepare_ml_dataset()
        
        print("\nFeature Statistics:")
        print(X.describe())
        
        print("\nTarget Distribution:")
        print(y.value_counts())


if __name__ == '__main__':
    main()

