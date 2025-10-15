"""
Snowflake Configuration Module

This module handles loading Snowflake connection parameters from environment variables.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class SnowflakeConfig:
    """Configuration class for Snowflake connection parameters."""
    
    def __init__(self):
        """Initialize Snowflake configuration from environment variables."""
        self.account = os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = os.getenv('SNOWFLAKE_USER')
        self.password = os.getenv('SNOWFLAKE_PASSWORD')
        self.warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
        self.database = os.getenv('SNOWFLAKE_DATABASE', 'ECOMMERCE_DB')
        self.schema = os.getenv('SNOWFLAKE_SCHEMA', 'ANALYTICS')
        self.role = os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
    
    def get_connection_params(self):
        """
        Get connection parameters as a dictionary.
        
        Returns:
            dict: Dictionary containing Snowflake connection parameters
        """
        return {
            'account': self.account,
            'user': self.user,
            'password': self.password,
            'warehouse': self.warehouse,
            'database': self.database,
            'schema': self.schema,
            'role': self.role
        }
    
    def validate(self):
        """
        Validate that all required configuration parameters are set.
        
        Raises:
            ValueError: If any required parameter is missing
        """
        required_params = ['account', 'user', 'password']
        missing_params = [param for param in required_params 
                         if not getattr(self, param)]
        
        if missing_params:
            raise ValueError(
                f"Missing required Snowflake configuration parameters: {', '.join(missing_params)}\n"
                f"Please set them in your .env file or environment variables."
            )
    
    def __repr__(self):
        """String representation of config (hiding password)."""
        return (f"SnowflakeConfig(account={self.account}, user={self.user}, "
                f"warehouse={self.warehouse}, database={self.database}, "
                f"schema={self.schema}, role={self.role})")


# Create a global config instance
config = SnowflakeConfig()

