"""
Credit Risk Modeling & Default Prediction
Author: Ravikumar Nalawade
Description: Random Forest Classifier to predict loan defaults while handling 
             class imbalance, preventing data leakage, and fixing timezone errors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- CONFIGURATION ---
# REPLACE THIS WITH YOUR ACTUAL FILE NAME
DATASET_PATH = '/Users/ravikumarnalawade/Documents/Certifications/Project/GitHub/Credit-Risk-Analysis/data/lcDataSample.csv'  
TARGET_COL = 'loan_status'
LEAKAGE_COLS = [
    'recoveries', 'collection_recovery_fee', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'next_pymnt_d', 'debt_settlement_flag'
]
DROP_COLS = ['id', 'member_id', 'url', 'desc', 'policy_code']

def load_and_clean_data(filepath):
    """Loads data, defines target, removes leakage, and engineers features."""
    print("Loading dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    
    # 1. Define Target: Only keep completed loans
    print("Filtering target classes...")
    target_mask = df[TARGET_COL].isin(['Fully Paid', 'Charged Off', 'Default'])
    df_clean = df[target_mask].copy()
    
    # Encode: 1 = Bad Loan (Default), 0 = Good Loan (Paid)
    df_clean['target'] = df_clean[TARGET_COL].apply(
        lambda x: 1 if x in ['Charged Off', 'Default'] else 0
    )
    
    # 2. Remove Data Leakage & Identifiers
    print("Removing data leakage columns...")
    cols_to_drop = LEAKAGE_COLS + DROP_COLS + [TARGET_COL]
    existing_drops = [c for c in cols_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=existing_drops)
    
    # 3. Feature Engineering: Dates & History
    print("Engineering features...")
    
    # --- TIMEZONE FIX IS HERE ---
    date_cols = ['issue_d', 'earliest_cr_line']
    for col in date_cols:
        if col in df_clean.columns:
            # Convert to datetime
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            # FORCE removal of timezone info so they can be subtracted
            df_clean[col] = df_clean[col].dt.tz_localize(None)

    # Credit History Length (Years)
    if 'issue_d' in df_clean.columns and 'earliest_cr_line' in df_clean.columns:
        df_clean['credit_hist_years'] = (df_clean['issue_d'] - df_clean['earliest_cr_line']).dt.days / 365
    
    # Drop original date columns as RF can't handle datetime objects directly
    df_clean = df_clean.drop(columns=date_cols, errors='ignore')
    
    return df_clean

def preprocess_and_train(df):
    """Handles missing values, encoding, and model training."""
    
    # --- FIX 1: Drop columns that are 100% empty (all NaNs) ---
    print("Dropping completely empty columns...")
    df = df.dropna(axis=1, how='all')
    
    # Select Features
    # Automatically select numeric types
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_features: numeric_features.remove('target')
    
    # Select categorical features (limit to those with < 50 unique values)
    categorical_features = [col for col in df.select_dtypes(include=['object', 'bool']).columns 
                            if df[col].nunique() < 50]
    
    print(f"Modeling with {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")
    
    X = df[numeric_features + categorical_features].copy()
    y = df['target']
    
    # Impute Numeric (Median)
    # verbose=0 suppresses warnings about skipped features
    imputer_num = SimpleImputer(strategy='median')
    X_num_imputed = imputer_num.fit_transform(X[numeric_features])
    
    # Encode Categorical (OneHot)
    # --- FIX 2: FORCE TO STRING ---
    # We convert everything to string so True becomes "True" and avoids the bool/str error
    X_cat = X[categorical_features].fillna('Missing').astype(str)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded_cat = encoder.fit_transform(X_cat)
    
    # Reassemble DataFrame
    cat_feature_names = encoder.get_feature_names_out(categorical_features)
    X_final = pd.DataFrame(
        data=np.hstack([X_num_imputed, X_encoded_cat]),
        columns=numeric_features + list(cat_feature_names)
    )
    
    # Split Data (Stratify is crucial for imbalanced data)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("Training Random Forest (this may take a moment)...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    return rf, X_train, X_test, y_test

def evaluate_model(model, X_test, y_test, threshold=0.2):
    """Evaluates model with specific focus on Recall and Threshold Tuning."""
    print("\n--- Model Evaluation ---")
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC-AUC
    roc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc:.4f}")
    
    # Apply Custom Threshold
    y_pred_new = (y_prob >= threshold).astype(int)
    
    print(f"\nConfusion Matrix (Threshold > {threshold}):")
    cm = confusion_matrix(y_test, y_pred_new)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_new))
    
    return cm

def plot_feature_importance(model, feature_names):
    """Plots and saves top 10 important features."""
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Important Features (Credit Risk)")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig('Feature_Importance.png')
    print("\nFeature Importance plot saved as 'Feature_Importance.png'")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # 1. Load & Clean
        df_clean = load_and_clean_data(DATASET_PATH)
        
        # 2. Train
        rf_model, X_train, X_test, y_test = preprocess_and_train(df_clean)
        
        # 3. Evaluate (Optimized Threshold 0.20)
        evaluate_model(rf_model, X_test, y_test, threshold=0.20)
        
        # 4. Visualize
        plot_feature_importance(rf_model, X_train.columns)
        
        print("\n✅ Process Complete. Ready for GitHub upload.")
        
    except FileNotFoundError:
        print(f"❌ Error: The file '{DATASET_PATH}' was not found. Please update line 16.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")