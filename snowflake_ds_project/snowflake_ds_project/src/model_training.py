"""
Model Training Module

This module handles machine learning model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging
from datetime import datetime
import json

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from .snowflake_connector import SnowflakeConnector
from .feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Class for training and evaluating churn prediction models.
    """
    
    def __init__(self, connector: SnowflakeConnector):
        """
        Initialize the ChurnPredictor.
        
        Args:
            connector: SnowflakeConnector instance
        """
        self.connector = connector
        self.engineer = FeatureEngineer(connector)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
    def prepare_data(self, test_size: float = 0.2, 
                     balance_data: bool = True) -> Tuple:
        """
        Prepare data for model training.
        
        Args:
            test_size: Proportion of data for testing
            balance_data: Whether to use SMOTE for balancing classes
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for model training...")
        
        # Get features from Snowflake
        X, y, feature_names = self.engineer.prepare_ml_dataset()
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Training set churn rate: {y_train.mean():.2%}")
        logger.info(f"Test set churn rate: {y_test.mean():.2%}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Balance data using SMOTE if requested
        if balance_data:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            logger.info(f"After SMOTE - Training set size: {len(X_train_scaled)}")
            logger.info(f"After SMOTE - Churn rate: {y_train.mean():.2%}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, model_type: str = 'xgboost', 
                   X_train: np.ndarray = None, 
                   y_train: np.ndarray = None,
                   tune_hyperparameters: bool = False) -> None:
        """
        Train a machine learning model.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'gradient_boosting', 'xgboost')
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        logger.info(f"Training {model_type} model...")
        
        if model_type == 'logistic':
            if tune_hyperparameters:
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
                base_model = LogisticRegression(random_state=42, max_iter=1000)
                self.model = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc')
            else:
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                
        elif model_type == 'random_forest':
            if tune_hyperparameters:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
                base_model = RandomForestClassifier(random_state=42)
                self.model = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc')
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                
        elif model_type == 'gradient_boosting':
            if tune_hyperparameters:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
                base_model = GradientBoostingClassifier(random_state=42)
                self.model = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc')
            else:
                self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                
        elif model_type == 'xgboost':
            if tune_hyperparameters:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
                base_model = XGBClassifier(random_state=42, eval_metric='logloss')
                self.model = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc')
            else:
                self.model = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    eval_metric='logloss'
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        if tune_hyperparameters:
            logger.info(f"Best parameters: {self.model.best_params_}")
            logger.info(f"Best cross-validation score: {self.model.best_score_:.4f}")
        
        logger.info("Model training completed")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall:    {self.metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {self.metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:   {self.metrics['roc_auc']:.4f}")
        logger.info("\nConfusion Matrix:")
        logger.info(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        logger.info(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
        logger.info("=" * 60)
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'best_estimator_'):
            if hasattr(self.model.best_estimator_, 'feature_importances_'):
                importances = self.model.best_estimator_.feature_importances_
            else:
                importances = np.abs(self.model.best_estimator_.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_model_to_snowflake(self, model_name: str) -> str:
        """
        Save model metadata to Snowflake.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model ID
        """
        logger.info("Saving model metadata to Snowflake...")
        
        model_id = f"MODEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get feature importance
        feature_importance_df = self.get_feature_importance()
        if feature_importance_df is not None:
            feature_importance_dict = dict(zip(
                feature_importance_df['feature'].tolist(),
                feature_importance_df['importance'].tolist()
            ))
        else:
            feature_importance_dict = {}
        
        # Prepare data for insertion
        insert_query = f"""
        INSERT INTO ECOMMERCE_DB.ML_MODELS.MODEL_METADATA
        (model_id, model_name, model_type, training_date, 
         accuracy, precision_score, recall_score, f1_score, auc_roc,
         feature_importance, hyperparameters)
        VALUES (
            '{model_id}',
            '{model_name}',
            '{type(self.model).__name__}',
            CURRENT_TIMESTAMP(),
            {self.metrics['accuracy']},
            {self.metrics['precision']},
            {self.metrics['recall']},
            {self.metrics['f1_score']},
            {self.metrics['roc_auc']},
            PARSE_JSON('{json.dumps(feature_importance_dict)}'),
            PARSE_JSON('{{}}')
        )
        """
        
        self.connector.execute_query(insert_query)
        logger.info(f"Model metadata saved with ID: {model_id}")
        
        return model_id
    
    def predict_and_save(self, model_id: str) -> None:
        """
        Generate predictions for all customers and save to Snowflake.
        
        Args:
            model_id: Model ID to associate with predictions
        """
        logger.info("Generating predictions for all customers...")
        
        # Get all features
        X, y, _ = self.engineer.prepare_ml_dataset()
        
        # Get customer IDs
        customer_ids_query = """
        SELECT customer_id
        FROM ECOMMERCE_DB.FEATURES.CUSTOMER_FEATURES
        ORDER BY customer_id
        """
        customer_ids_df = self.connector.execute_query_to_df(customer_ids_query)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        predictions = self.model.predict(X_scaled)
        prediction_probas = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'prediction_id': [f"PRED_{i+1:08d}" for i in range(len(predictions))],
            'model_id': model_id,
            'customer_id': customer_ids_df['CUSTOMER_ID'].values,
            'prediction_date': datetime.now(),
            'churn_probability': prediction_probas,
            'predicted_churn': predictions
        })
        
        # Save to Snowflake
        self.connector.load_dataframe(
            df=predictions_df,
            table_name='PREDICTIONS',
            schema='ML_MODELS',
            if_exists='append'
        )
        
        logger.info(f"Saved {len(predictions_df)} predictions to Snowflake")


def main():
    """Main function to demonstrate model training pipeline."""
    from config import config
    
    # Validate configuration
    config.validate()
    
    # Connect to Snowflake
    with SnowflakeConnector(config.get_connection_params()) as connector:
        # Create predictor
        predictor = ChurnPredictor(connector)
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(
            test_size=0.2,
            balance_data=True
        )
        
        # Train model
        predictor.train_model(
            model_type='xgboost',
            X_train=X_train,
            y_train=y_train,
            tune_hyperparameters=False
        )
        
        # Evaluate model
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # Get feature importance
        feature_importance = predictor.get_feature_importance()
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Save model to Snowflake
        model_id = predictor.save_model_to_snowflake('XGBoost_Churn_Predictor')
        
        # Generate and save predictions
        predictor.predict_and_save(model_id)


if __name__ == '__main__':
    main()

