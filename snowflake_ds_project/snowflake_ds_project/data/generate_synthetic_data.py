"""
Synthetic Data Generation for E-Commerce Churn Prediction

This script generates realistic synthetic data for customers, transactions, and activity.
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)


def generate_customers(n_customers: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic customer data.
    
    Args:
        n_customers: Number of customers to generate
        
    Returns:
        DataFrame with customer data
    """
    print(f"Generating {n_customers} customers...")
    
    customers = []
    start_date = datetime.now() - timedelta(days=730)  # 2 years ago
    
    for i in range(n_customers):
        customer = {
            'customer_id': f'CUST_{i+1:06d}',
            'registration_date': fake.date_between(start_date=start_date, end_date='today'),
            'age': np.random.randint(18, 75),
            'gender': np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04]),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia'], 
                                       p=[0.40, 0.20, 0.15, 0.10, 0.10, 0.05]),
            'membership_tier': np.random.choice(['Bronze', 'Silver', 'Gold'], 
                                               p=[0.60, 0.30, 0.10])
        }
        customers.append(customer)
    
    df = pd.DataFrame(customers)
    print(f"Generated {len(df)} customers")
    return df


def generate_transactions(customers_df: pd.DataFrame, 
                         avg_transactions_per_customer: int = 15) -> pd.DataFrame:
    """
    Generate synthetic transaction data.
    
    Args:
        customers_df: DataFrame with customer data
        avg_transactions_per_customer: Average number of transactions per customer
        
    Returns:
        DataFrame with transaction data
    """
    print(f"Generating transactions...")
    
    transactions = []
    transaction_id = 1
    
    product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 
                         'Books', 'Beauty', 'Toys', 'Food & Beverage']
    
    for _, customer in customers_df.iterrows():
        # Determine if customer will churn (30% churn rate)
        will_churn = np.random.random() < 0.30
        
        # Number of transactions for this customer
        n_transactions = np.random.poisson(avg_transactions_per_customer)
        n_transactions = max(1, n_transactions)  # At least 1 transaction
        
        # Generate transactions
        registration_date = pd.to_datetime(customer['registration_date'])
        
        for i in range(n_transactions):
            # Transaction date between registration and now
            if will_churn and i >= n_transactions - 3:
                # Last few transactions are older (more than 90 days ago)
                max_date = datetime.now() - timedelta(days=100)
            else:
                max_date = datetime.now()
            
            days_range = (max_date - registration_date).days
            if days_range <= 0:
                days_range = 1
                
            transaction_date = registration_date + timedelta(days=np.random.randint(0, days_range))
            
            # Amount varies by membership tier
            tier_multiplier = {'Bronze': 1.0, 'Silver': 1.5, 'Gold': 2.5}
            base_amount = np.random.gamma(shape=2, scale=30)
            amount = base_amount * tier_multiplier[customer['membership_tier']]
            
            transaction = {
                'transaction_id': f'TXN_{transaction_id:08d}',
                'customer_id': customer['customer_id'],
                'transaction_date': transaction_date,
                'amount': round(amount, 2),
                'product_category': np.random.choice(product_categories),
                'quantity': np.random.randint(1, 6)
            }
            transactions.append(transaction)
            transaction_id += 1
    
    df = pd.DataFrame(transactions)
    print(f"Generated {len(df)} transactions")
    return df


def generate_customer_activity(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic customer activity data.
    
    Args:
        customers_df: DataFrame with customer data
        
    Returns:
        DataFrame with customer activity data
    """
    print(f"Generating customer activity...")
    
    activities = []
    
    for i, customer in customers_df.iterrows():
        # Activity correlates with engagement
        engagement_level = np.random.choice(['low', 'medium', 'high'], 
                                           p=[0.30, 0.50, 0.20])
        
        if engagement_level == 'low':
            page_views = np.random.randint(0, 50)
            support_tickets = np.random.randint(0, 3)
            email_opened = np.random.randint(0, 5)
            last_login_days_ago = np.random.randint(30, 180)
        elif engagement_level == 'medium':
            page_views = np.random.randint(50, 200)
            support_tickets = np.random.randint(0, 5)
            email_opened = np.random.randint(5, 20)
            last_login_days_ago = np.random.randint(7, 60)
        else:  # high
            page_views = np.random.randint(200, 1000)
            support_tickets = np.random.randint(0, 8)
            email_opened = np.random.randint(20, 50)
            last_login_days_ago = np.random.randint(0, 14)
        
        activity = {
            'activity_id': f'ACT_{i+1:06d}',
            'customer_id': customer['customer_id'],
            'last_login_date': datetime.now() - timedelta(days=last_login_days_ago),
            'page_views': page_views,
            'support_tickets': support_tickets,
            'email_opened': email_opened
        }
        activities.append(activity)
    
    df = pd.DataFrame(activities)
    print(f"Generated {len(df)} activity records")
    return df


def main():
    """Generate all synthetic datasets and save to CSV files."""
    print("=" * 60)
    print("Synthetic Data Generation for E-Commerce Churn Prediction")
    print("=" * 60)
    
    # Generate data
    customers_df = generate_customers(n_customers=10000)
    transactions_df = generate_transactions(customers_df, avg_transactions_per_customer=15)
    activity_df = generate_customer_activity(customers_df)
    
    # Save to CSV
    output_dir = '/home/ubuntu/snowflake_ds_project/data'
    
    customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
    print(f"\nSaved customers data to {output_dir}/customers.csv")
    
    transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
    print(f"Saved transactions data to {output_dir}/transactions.csv")
    
    activity_df.to_csv(f'{output_dir}/customer_activity.csv', index=False)
    print(f"Saved activity data to {output_dir}/customer_activity.csv")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Data Generation Summary")
    print("=" * 60)
    print(f"Total Customers: {len(customers_df):,}")
    print(f"Total Transactions: {len(transactions_df):,}")
    print(f"Total Activity Records: {len(activity_df):,}")
    print(f"\nAverage Transactions per Customer: {len(transactions_df) / len(customers_df):.2f}")
    print(f"Total Transaction Value: ${transactions_df['amount'].sum():,.2f}")
    print(f"Average Transaction Value: ${transactions_df['amount'].mean():.2f}")
    
    print("\nMembership Distribution:")
    print(customers_df['membership_tier'].value_counts())
    
    print("\nCountry Distribution:")
    print(customers_df['country'].value_counts())


if __name__ == '__main__':
    main()

