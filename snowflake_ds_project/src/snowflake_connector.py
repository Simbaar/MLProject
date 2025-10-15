"""
Snowflake Connector Module

This module provides a wrapper class for connecting to Snowflake and executing queries.
"""

import snowflake.connector
from snowflake.connector import DictCursor
import pandas as pd
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowflakeConnector:
    """
    A wrapper class for Snowflake database connections.
    
    This class provides methods for connecting to Snowflake, executing queries,
    and loading data into Snowflake tables.
    """
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize the Snowflake connector.
        
        Args:
            config: Dictionary containing Snowflake connection parameters
        """
        self.config = config
        self.connection = None
        self.cursor = None
    
    def connect(self) -> None:
        """
        Establish a connection to Snowflake.
        
        Raises:
            Exception: If connection fails
        """
        try:
            self.connection = snowflake.connector.connect(
                account=self.config['account'],
                user=self.config['user'],
                password=self.config['password'],
                warehouse=self.config['warehouse'],
                database=self.config['database'],
                schema=self.config['schema'],
                role=self.config['role']
            )
            self.cursor = self.connection.cursor(DictCursor)
            logger.info(f"Successfully connected to Snowflake account: {self.config['account']}")
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Close the Snowflake connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Snowflake")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            List of dictionaries containing query results
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            results = self.cursor.fetchall()
            logger.info(f"Query executed successfully. Rows returned: {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def execute_query_to_df(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            pandas DataFrame containing query results
        """
        try:
            df = pd.read_sql(query, self.connection)
            logger.info(f"Query executed successfully. DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def execute_sql_file(self, file_path: str) -> None:
        """
        Execute SQL commands from a file.
        
        Args:
            file_path: Path to SQL file
        """
        try:
            with open(file_path, 'r') as file:
                sql_commands = file.read()
            
            # Split by semicolon and execute each command
            for command in sql_commands.split(';'):
                command = command.strip()
                if command:
                    self.cursor.execute(command)
            
            logger.info(f"Successfully executed SQL file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to execute SQL file: {str(e)}")
            raise
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, 
                       schema: Optional[str] = None, 
                       if_exists: str = 'append') -> None:
        """
        Load a pandas DataFrame into a Snowflake table.
        
        Args:
            df: pandas DataFrame to load
            table_name: Target table name
            schema: Optional schema name (uses connection default if not provided)
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
        """
        try:
            from snowflake.connector.pandas_tools import write_pandas
            
            # Set schema if provided
            if schema:
                full_table_name = f"{schema}.{table_name}"
            else:
                full_table_name = table_name
            
            success, nchunks, nrows, _ = write_pandas(
                conn=self.connection,
                df=df,
                table_name=table_name.upper(),
                schema=schema.upper() if schema else self.config['schema'].upper(),
                auto_create_table=True,
                overwrite=(if_exists == 'replace')
            )
            
            if success:
                logger.info(f"Successfully loaded {nrows} rows into {full_table_name}")
            else:
                logger.warning(f"Load operation completed with warnings for {full_table_name}")
                
        except Exception as e:
            logger.error(f"Failed to load DataFrame into Snowflake: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str, schema: Optional[str] = None) -> pd.DataFrame:
        """
        Get information about a table's structure.
        
        Args:
            table_name: Name of the table
            schema: Optional schema name
            
        Returns:
            DataFrame with table column information
        """
        schema_name = schema if schema else self.config['schema']
        query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM {self.config['database']}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = '{schema_name.upper()}'
          AND table_name = '{table_name.upper()}'
        ORDER BY ordinal_position
        """
        return self.execute_query_to_df(query)
    
    def get_row_count(self, table_name: str, schema: Optional[str] = None) -> int:
        """
        Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
            schema: Optional schema name
            
        Returns:
            Number of rows in the table
        """
        schema_name = schema if schema else self.config['schema']
        query = f"SELECT COUNT(*) as count FROM {schema_name}.{table_name}"
        result = self.execute_query(query)
        return result[0]['COUNT']
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

