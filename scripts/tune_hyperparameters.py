"""
CLI Script for Hyperparameter Tuning
HEA - NASA Space Apps Challenge 2025

This script provides command-line interface for hyperparameter optimization
of the exoplanet detection models.

Usage:
    python scripts/tune_hyperparameters.py --config config/tuning_config.yaml
    python scripts/tune_hyperparameters.py --quick --trials 50
    python scripts/tune_hyperparameters.py --model rf --trials 100
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.hyperparameter_tuning import HyperparameterTuner

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def load_training_data(data_path, target_column='LABEL'):
    """Load and prepare training data"""
    try:
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            print(f"Warning: Target column '{target_column}' not found. Using last column as target.")
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
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
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Data loaded: {len(X)} samples, {len(X.columns)} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def run_hyperparameter_tuning(args):
    """Main hyperparameter tuning function"""
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/cli_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting HEA Hyperparameter Tuning")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
        if config is None:
            logger.error("Failed to load configuration file")
            return False
    
    # Set parameters from args or config
    data_path = args.data or config.get('data_path', 'data/processed/training_data.csv')
    n_trials = args.trials or config.get('n_trials', 100)
    test_size = args.test_size or config.get('test_size', 0.2)
    models = args.models.split(',') if args.models else config.get('models', ['rf', 'gb', 'xgb', 'lgbm'])
    output_dir = args.output or config.get('output_dir', 'models/hyperparameter_results')
    
    logger.info(f"Configuration:")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Test size: {test_size}")
    logger.info(f"  Models: {models}")
    logger.info(f"  Output dir: {output_dir}")
    
    # Load data
    X, y = load_training_data(data_path)
    if X is None or y is None:
        logger.error("Failed to load training data")
        return False
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Validation={len(X_val)}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        n_trials=n_trials
    )
    
    try:
        if args.single_model:
            # Optimize single model
            logger.info(f"Optimizing single model: {args.single_model}")
            best_params, best_score = tuner.optimize_single_model(args.single_model)
            
            results = {
                'model_type': args.single_model,
                'best_params': best_params,
                'best_score': best_score,
                'feature_names': X.columns.tolist()
            }
            
        else:
            # Optimize ensemble
            logger.info("Optimizing ensemble model")
            results = tuner.optimize_ensemble(model_types=models)
        
        # Save results
        model_path, params_path = tuner.save_results(results, output_dir)
        
        logger.info("="*50)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*50)
        
        if args.single_model:
            logger.info(f"Best {args.single_model} Score: {results['best_score']:.4f}")
            logger.info(f"Best Parameters: {results['best_params']}")
        else:
            logger.info("Final Ensemble Metrics:")
            for metric, value in results['metrics'].items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info(f"Optimal Threshold: {results['optimal_threshold']:.3f}")
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Parameters saved to: {params_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='HEA Hyperparameter Tuning CLI')
    
    # Data arguments
    parser.add_argument('--data', type=str, help='Path to training data CSV file')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    
    # Tuning arguments
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation split size')
    parser.add_argument('--models', type=str, help='Comma-separated list of models (rf,gb,xgb,lgbm)')
    parser.add_argument('--single-model', type=str, choices=['rf', 'gb', 'xgb', 'lgbm'], 
                       help='Optimize single model instead of ensemble')
    
    # Output arguments
    parser.add_argument('--output', type=str, help='Output directory for results')
    
    # Convenience arguments
    parser.add_argument('--quick', action='store_true', help='Quick tuning with 50 trials')
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.trials = 50
        print("Quick tuning mode: Using 50 trials")
    
    # Validate arguments
    if not args.data and not args.config:
        print("Error: Must specify either --data or --config")
        parser.print_help()
        return 1
    
    # Run tuning
    success = run_hyperparameter_tuning(args)
    
    if success:
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
        print("="*50)
        return 0
    else:
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING FAILED!")
        print("="*50)
        return 1

if __name__ == "__main__":
    exit(main())