"""
Data Loader Module

This module handles loading data from CSV files into Snowflake tables.
"""

import pandas as pd
from typing import Dict
import logging
from pathlib import Path
from .snowflake_connector import SnowflakeConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading data into Snowflake from various sources.
    """
    
    def __init__(self, connector: SnowflakeConnector):
        """
        Initialize the DataLoader.
        
        Args:
            connector: SnowflakeConnector instance
        """
        self.connector = connector
    
    def load_csv_to_snowflake(self, csv_path: str, table_name: str, 
                               schema: str = 'RAW_DATA', 
                               if_exists: str = 'replace') -> None:
        """
        Load data from CSV file into Snowflake table.
        
        Args:
            csv_path: Path to CSV file
            table_name: Target table name in Snowflake
            schema: Target schema name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
        """
        try:
            logger.info(f"Reading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            
            logger.info(f"Loaded {len(df)} rows from CSV")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Load into Snowflake
            self.connector.load_dataframe(
                df=df,
                table_name=table_name,
                schema=schema,
                if_exists=if_exists
            )
            
            logger.info(f"Successfully loaded data into {schema}.{table_name}")
            
        except Exception as e:
            logger.error(f"Failed to load CSV to Snowflake: {str(e)}")
            raise
    
    def load_all_data(self, data_dir: str = '/home/ubuntu/snowflake_ds_project/data') -> None:
        """
        Load all CSV files from data directory into Snowflake.
        
        Args:
            data_dir: Directory containing CSV files
        """
        data_path = Path(data_dir)
        
        # Define mapping of CSV files to table names
        file_table_mapping = {
            'customers.csv': 'CUSTOMERS',
            'transactions.csv': 'TRANSACTIONS',
            'customer_activity.csv': 'CUSTOMER_ACTIVITY'
        }
        
        logger.info("=" * 60)
        logger.info("Starting data load process")
        logger.info("=" * 60)
        
        for csv_file, table_name in file_table_mapping.items():
            csv_path = data_path / csv_file
            
            if csv_path.exists():
                logger.info(f"\nLoading {csv_file} -> {table_name}")
                self.load_csv_to_snowflake(
                    csv_path=str(csv_path),
                    table_name=table_name,
                    schema='RAW_DATA',
                    if_exists='replace'
                )
            else:
                logger.warning(f"CSV file not found: {csv_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Data load process completed")
        logger.info("=" * 60)
    
    def verify_data_load(self) -> Dict[str, int]:
        """
        Verify that data was loaded successfully by checking row counts.
        
        Returns:
            Dictionary with table names and row counts
        """
        tables = ['CUSTOMERS', 'TRANSACTIONS', 'CUSTOMER_ACTIVITY']
        row_counts = {}
        
        logger.info("\nVerifying data load...")
        logger.info("-" * 40)
        
        for table in tables:
            try:
                count = self.connector.get_row_count(table, schema='RAW_DATA')
                row_counts[table] = count
                logger.info(f"{table}: {count:,} rows")
            except Exception as e:
                logger.error(f"Failed to get row count for {table}: {str(e)}")
                row_counts[table] = -1
        
        return row_counts
    
    def create_feature_table(self) -> None:
        """
        Create the feature table by executing the feature engineering query.
        """
        logger.info("\nCreating feature table...")
        
        query = """
        CREATE OR REPLACE TABLE ECOMMERCE_DB.FEATURES.CUSTOMER_FEATURES AS
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
        FROM ECOMMERCE_DB.ANALYTICS.CUSTOMER_360
        """
        
        try:
            self.connector.execute_query(query)
            count = self.connector.get_row_count('CUSTOMER_FEATURES', schema='FEATURES')
            logger.info(f"Feature table created successfully with {count:,} rows")
        except Exception as e:
            logger.error(f"Failed to create feature table: {str(e)}")
            raise


def main():
    """Main function to demonstrate data loading."""
    from config import config
    
    # Validate configuration
    config.validate()
    
    # Connect to Snowflake
    with SnowflakeConnector(config.get_connection_params()) as connector:
        # Create data loader
        loader = DataLoader(connector)
        
        # Load all data
        loader.load_all_data()
        
        # Verify data load
        row_counts = loader.verify_data_load()
        
        # Create feature table
        loader.create_feature_table()


if __name__ == '__main__':
    main()

