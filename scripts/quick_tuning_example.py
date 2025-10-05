"""
Simple Example: Quick Hyperparameter Tuning
HEA - NASA Space Apps Challenge 2025

This script demonstrates the simplest way to use hyperparameter tuning
with your exoplanet detection data.

Usage:
    python scripts/quick_tuning_example.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.hyperparameter_tuning import quick_tune_model
    print(" Hyperparameter tuning module available")
except ImportError as e:
    print(f" Hyperparameter tuning not available: {e}")
    print("Install dependencies: pip install optuna xgboost lightgbm")
    print(f"Project root: {project_root}")
    print(f"Looking for: {project_root / 'src' / 'hyperparameter_tuning.py'}")
    sys.exit(1)

def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample exoplanet data...")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some signal
    signal = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3) > 0.5
    noise = np.random.randn(n_samples) * 0.1
    y = (signal + noise > 0.5).astype(int)
    
    # Convert to DataFrame
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='LABEL')
    
    print(f"Sample data created: {len(X_df)} samples, {len(X_df.columns)} features")
    print(f"Class distribution: {y_series.value_counts().to_dict()}")
    
    return X_df, y_series

def main():
    print("="*60)
    print("HEA QUICK HYPERPARAMETER TUNING EXAMPLE")
    print("="*60)
    
    # Create or load data
    X, y = create_sample_data()
    
    # Split data for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Data split: {len(X_train)} training, {len(X_val)} validation")
    
    # Quick tuning (50 trials)
    print("\nStarting quick hyperparameter optimization...")
    print("This will optimize Random Forest, Gradient Boosting, and ensemble models")
    
    try:
        results = quick_tune_model(
            X_train=X_train,
            X_val=X_val, 
            y_train=y_train,
            y_val=y_val,
            n_trials=50  # Quick optimization
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        # Display metrics
        metrics = results['metrics']
        print(f"Final Ensemble Performance:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%") 
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:  {metrics['f1_score']*100:.2f}%")
        print(f"  AUC:       {metrics['auc_score']:.4f}")
        print(f"  Optimal Threshold: {results['optimal_threshold']:.3f}")
        
        # Show optimized parameters
        print(f"\nOptimized Models:")
        for model_type, params in results['optimized_params'].items():
            print(f"  {model_type.upper()}: {len(params)} parameters optimized")
        
        print("\n" + "="*60)
        print("SUCCESS! Hyperparameter optimization completed.")
        print("="*60)
        print("You can now:")
        print("1. Use results['ensemble'] as your trained model")
        print("2. Apply results['optimal_threshold'] for predictions")
        print("3. Integrate into your existing pipeline")
        
        # Quick test
        print(f"\nQuick test on validation data:")
        y_pred_proba = results['ensemble'].predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= results['optimal_threshold']).astype(int)
        
        from sklearn.metrics import accuracy_score
        test_accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation accuracy: {test_accuracy*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[CELEBRATE] Example completed successfully!")
        print("Try the webapp interface or CLI tools for more advanced features.")
    else:
        print("\n[FAIL] Example failed.")
        print("Check dependencies and data format.")