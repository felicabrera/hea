"""
Enhanced Training Script with Hyperparameter Tuning
HEA - NASA Space Apps Challenge 2025

This script demonstrates how to train models with hyperparameter optimization
and integrate the results into the existing model pipeline.

Usage:
    python scripts/train_with_tuning.py --data data/processed/training_data.csv
    python scripts/train_with_tuning.py --quick
    python scripts/train_with_tuning.py --config config/tuning_config.yaml
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.hyperparameter_tuning import HyperparameterTuner, quick_tune_model

def load_and_prepare_data(data_path, target_column='LABEL'):
    """Load and prepare training data"""
    print(f"Loading data from: {data_path}")
    
    # Check if file exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        return None, None
    
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            print(f"Warning: Target column '{target_column}' not found. Using last column as target.")
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
        print(f"Features: {len(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def preprocess_features(X):
    """Preprocess features for model training"""
    print("Preprocessing features...")
    
    # Convert string columns to numeric
    string_cols = X.select_dtypes(include=['object']).columns
    if len(string_cols) > 0:
        print(f"Converting {len(string_cols)} string columns to numeric...")
        for col in string_cols:
            if 'rastr' in col.lower() or col == 'rastr':
                # Special handling for RA string format
                def convert_ra(ra_str):
                    try:
                        if pd.isna(ra_str) or ra_str == '':
                            return 0.0
                        if isinstance(ra_str, (int, float)):
                            return float(ra_str)
                        parts = str(ra_str).split(':')
                        hours = float(parts[0])
                        minutes = float(parts[1]) if len(parts) > 1 else 0
                        seconds = float(parts[2]) if len(parts) > 2 else 0
                        return (hours + minutes/60 + seconds/3600) * 15
                    except:
                        return 0.0
                X[col] = X[col].apply(convert_ra)
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Handle missing values and infinities
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"Filling {nan_count} missing values with zeros")
        X = X.fillna(0)
    
    inf_count = np.isinf(X.values).sum()
    if inf_count > 0:
        print(f"Replacing {inf_count} infinite values with zeros")
        X = X.replace([np.inf, -np.inf], 0)
    
    print(f"Final feature matrix: {X.shape}")
    return X

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model performance"""
    print(f"\nEvaluating model with threshold {threshold:.3f}...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return y_pred, y_pred_proba

def save_model_for_webapp(results, output_path="models/catalog_models"):
    """Save optimized model in webapp-compatible format"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ultra_model_optimized_{timestamp}.joblib"
    filepath = output_dir / filename
    
    # Create webapp-compatible model data structure
    model_data = {
        'stacking': results['ensemble'],
        'feature_names': results.get('feature_names', []),
        'test_accuracy': results['metrics']['accuracy'],
        'test_precision': results['metrics']['precision'],
        'test_recall': results['metrics']['recall'],
        'test_f1': results['metrics']['f1_score'],
        'test_auc': results['metrics']['auc_score'],
        'optimal_threshold': results['optimal_threshold'],
        'training_timestamp': timestamp,
        'optimization_params': results['optimized_params']
    }
    
    # Save model
    joblib.dump(model_data, filepath)
    print(f"Model saved for webapp: {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Train HEA models with hyperparameter optimization')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='data/processed/training_data.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--target', type=str, default='LABEL',
                       help='Target column name')
    
    # Tuning arguments
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test split size')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Validation split size (from remaining data after test split)')
    
    # Model arguments
    parser.add_argument('--models', type=str, default='rf,gb,xgb,lgbm',
                       help='Comma-separated list of models to include')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='models/catalog_models',
                       help='Output directory for webapp-compatible model')
    
    # Convenience arguments
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with 50 trials')
    parser.add_argument('--config', type=str,
                       help='Configuration file (YAML)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HEA TRAINING WITH HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Handle quick mode
    if args.quick:
        args.trials = 50
        print("Quick mode: Using 50 trials")
    
    # Load and prepare data
    X, y = load_and_prepare_data(args.data, args.target)
    if X is None or y is None:
        print("Failed to load data. Exiting.")
        return 1
    
    # Preprocess features
    X = preprocess_features(X)
    
    # Split data: first split for test set, then split remaining for train/val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_size/(1-args.test_size), 
        random_state=42, stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples") 
    print(f"  Test: {len(X_test)} samples")
    
    # Parse model types
    model_types = [m.strip() for m in args.models.split(',')]
    print(f"  Model types: {model_types}")
    
    try:\n        print(\"\\n\" + \"=\"*60)\n        print(\"STARTING HYPERPARAMETER OPTIMIZATION\")\n        print(\"=\"*60)\n        \n        # Initialize tuner\n        tuner = HyperparameterTuner(\n            X_train=X_train,\n            X_val=X_val,\n            y_train=y_train,\n            y_val=y_val,\n            n_trials=args.trials\n        )\n        \n        # Run optimization\n        results = tuner.optimize_ensemble(model_types=model_types)\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"OPTIMIZATION RESULTS\")\n        print(\"=\"*60)\n        \n        # Display results\n        metrics = results['metrics']\n        print(f\"Validation Metrics:\")\n        print(f\"  Accuracy:  {metrics['accuracy']*100:.2f}%\")\n        print(f\"  Precision: {metrics['precision']*100:.2f}%\")\n        print(f\"  Recall:    {metrics['recall']*100:.2f}%\")\n        print(f\"  F1 Score:  {metrics['f1_score']*100:.2f}%\")\n        print(f\"  AUC:       {metrics['auc_score']:.4f}\")\n        print(f\"  Threshold: {results['optimal_threshold']:.3f}\")\n        \n        # Test set evaluation\n        print(\"\\n\" + \"=\"*60)\n        print(\"TEST SET EVALUATION\")\n        print(\"=\"*60)\n        \n        evaluate_model(results['ensemble'], X_test, y_test, results['optimal_threshold'])\n        \n        # Save results\n        print(\"\\n\" + \"=\"*60)\n        print(\"SAVING RESULTS\")\n        print(\"=\"*60)\n        \n        # Save hyperparameter results\n        tuner.save_results(results)\n        \n        # Save for webapp\n        webapp_path = save_model_for_webapp(results, args.output)\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"TRAINING COMPLETED SUCCESSFULLY!\")\n        print(\"=\"*60)\n        print(f\"Webapp model saved to: {webapp_path}\")\n        print(\"You can now load this model in the HEA webapp.\")\n        \n        return 0\n        \n    except Exception as e:\n        print(f\"\\nTraining failed: {str(e)}\")\n        return 1\n\nif __name__ == \"__main__\":\n    exit(main())"
  },
        {
            "explanation": "Complete enhanced training script with hyperparameter optimization",
            "oldString": "    try:\n        print(\"\\n\" + \"=\"*60)\n        print(\"STARTING HYPERPARAMETER OPTIMIZATION\")\n        print(\"=\"*60)\n        \n        # Initialize tuner\n        tuner = HyperparameterTuner(\n            X_train=X_train,\n            X_val=X_val,\n            y_train=y_train,\n            y_val=y_val,\n            n_trials=args.trials\n        )\n        \n        # Run optimization\n        results = tuner.optimize_ensemble(model_types=model_types)\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"OPTIMIZATION RESULTS\")\n        print(\"=\"*60)\n        \n        # Display results\n        metrics = results['metrics']\n        print(f\"Validation Metrics:\")\n        print(f\"  Accuracy:  {metrics['accuracy']*100:.2f}%\")\n        print(f\"  Precision: {metrics['precision']*100:.2f}%\")\n        print(f\"  Recall:    {metrics['recall']*100:.2f}%\")\n        print(f\"  F1 Score:  {metrics['f1_score']*100:.2f}%\")\n        print(f\"  AUC:       {metrics['auc_score']:.4f}\")\n        print(f\"  Threshold: {results['optimal_threshold']:.3f}\")\n        \n        # Test set evaluation\n        print(\"\\n\" + \"=\"*60)\n        print(\"TEST SET EVALUATION\")\n        print(\"=\"*60)\n        \n        evaluate_model(results['ensemble'], X_test, y_test, results['optimal_threshold'])\n        \n        # Save results\n        print(\"\\n\" + \"=\"*60)\n        print(\"SAVING RESULTS\")\n        print(\"=\"*60)\n        \n        # Save hyperparameter results\n        tuner.save_results(results)\n        \n        # Save for webapp\n        webapp_path = save_model_for_webapp(results, args.output)\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"TRAINING COMPLETED SUCCESSFULLY!\")\n        print(\"=\"*60)\n        print(f\"Webapp model saved to: {webapp_path}\")\n        print(\"You can now load this model in the HEA webapp.\")\n        \n        return 0\n        \n    except Exception as e:\n        print(f\"\\nTraining failed: {str(e)}\")\n        return 1\n\nif __name__ == \"__main__\":\n    exit(main())",
            "filePath": "C:\\Users\\feli\\Dev\\hea\\scripts\\train_with_tuning.py",
            "newString": "    try:\n        print(\"\\n\" + \"=\"*60)\n        print(\"STARTING HYPERPARAMETER OPTIMIZATION\")\n        print(\"=\"*60)\n        \n        # Initialize tuner\n        tuner = HyperparameterTuner(\n            X_train=X_train,\n            X_val=X_val,\n            y_train=y_train,\n            y_val=y_val,\n            n_trials=args.trials\n        )\n        \n        # Run optimization\n        results = tuner.optimize_ensemble(model_types=model_types)\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"OPTIMIZATION RESULTS\")\n        print(\"=\"*60)\n        \n        # Display results\n        metrics = results['metrics']\n        print(f\"Validation Metrics:\")\n        print(f\"  Accuracy:  {metrics['accuracy']*100:.2f}%\")\n        print(f\"  Precision: {metrics['precision']*100:.2f}%\")\n        print(f\"  Recall:    {metrics['recall']*100:.2f}%\")\n        print(f\"  F1 Score:  {metrics['f1_score']*100:.2f}%\")\n        print(f\"  AUC:       {metrics['auc_score']:.4f}\")\n        print(f\"  Threshold: {results['optimal_threshold']:.3f}\")\n        \n        # Test set evaluation\n        print(\"\\n\" + \"=\"*60)\n        print(\"TEST SET EVALUATION\")\n        print(\"=\"*60)\n        \n        evaluate_model(results['ensemble'], X_test, y_test, results['optimal_threshold'])\n        \n        # Save results\n        print(\"\\n\" + \"=\"*60)\n        print(\"SAVING RESULTS\")\n        print(\"=\"*60)\n        \n        # Save hyperparameter results\n        tuner.save_results(results)\n        \n        # Save for webapp\n        webapp_path = save_model_for_webapp(results, args.output)\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"TRAINING COMPLETED SUCCESSFULLY!\")\n        print(\"=\"*60)\n        print(f\"Webapp model saved to: {webapp_path}\")\n        print(\"You can now load this model in the HEA webapp.\")\n        \n        return 0\n        \n    except Exception as e:\n        print(f\"\\nTraining failed: {str(e)}\")\n        return 1\n\nif __name__ == \"__main__\":\n    exit(main())"
        }
    ]