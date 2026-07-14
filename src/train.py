# -*- coding: utf-8 -*-
"""
Model training script for Differentiated Thyroid Cancer Recurrence prediction.
Trains a Random Forest classifier excluding treatment response features to predict early recurrence.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model():
    print("Loading processed dataset...")
    # Path relative to project root / execution dir
    processed_data_path = 'data/processed/thyroid_processed.csv'
    if not os.path.exists(processed_data_path):
        # Try parent directory relative path for execution within src/
        processed_data_path = '../data/processed/thyroid_processed.csv'
        
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data file not found at {processed_data_path}. Please run preprocessing first.")
        
    df = pd.read_csv(processed_data_path)
    
    # Split features and target
    X = df.drop('Recurred', axis=1)
    y = df['Recurred']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Original train shape: {X_train.shape}")
    print(f"Original test shape: {X_test.shape}")
    
    # Drop Response_ columns (known post-treatment, not suitable for early prediction)
    response_cols = [c for c in X_train.columns if c.startswith('Response_')]
    X_train_no_response = X_train.drop(columns=response_cols)
    X_test_no_response = X_test.drop(columns=response_cols)
    
    print(f"Training features shape (excluding post-treatment columns): {X_train_no_response.shape}")
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier (balanced class weights)...")
    rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_model.fit(X_train_no_response, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test_no_response)
    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Ensure models directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save artifacts
    model_path = os.path.join(models_dir, 'best_model.pkl')
    features_path = os.path.join(models_dir, 'feature_columns.pkl')
    encoding_info_path = os.path.join(models_dir, 'encoding_info.pkl')
    
    joblib.dump(rf_model, model_path)
    print(f"Saved model to {model_path}")
    
    joblib.dump(X_train_no_response.columns.tolist(), features_path)
    print(f"Saved feature columns list to {features_path}")
    
    encoding_info = {
        'risk_order': ['Low', 'Intermediate', 'High'],
        't_order': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'],
        'n_order': ['N0', 'N1a', 'N1b'],
        'stage_order': ['I', 'II', 'III', 'IVA', 'IVB'],
        'adenopathy_map': {
            'No': 'No', 'Right': 'Unilateral', 'Left': 'Unilateral',
            'Bilateral': 'Bilateral_Extensive', 'Extensive': 'Bilateral_Extensive',
            'Posterior': 'Bilateral_Extensive'
        }
    }
    joblib.dump(encoding_info, encoding_info_path)
    print(f"Saved encoding info metadata to {encoding_info_path}")
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    train_model()
