"""
Hyperparameter Tuning Module for HEA
NASA Space Apps Challenge 2025

This module provides comprehensive hyperparameter optimization capabilities
for the ensemble machine learning models used in habitable exoplanet analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using alternative models.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Using alternative models.")

class HyperparameterTuner:
    """
    Advanced hyperparameter tuning system for exoplanet detection models
    """
    
    def __init__(self, X_train, X_val, y_train, y_val, n_trials=100, cv_folds=5):
        """
        Initialize the hyperparameter tuner
        
        Args:
            X_train: Training features
            X_val: Validation features  
            y_train: Training labels
            y_val: Validation labels
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
        """
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        
        # Results storage
        self.best_params = {}
        self.best_scores = {}
        self.optimization_history = []
        self.study = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for tuning process"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/hyperparameter_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_model_parameter_space(self, model_type):
        """
        Define parameter search spaces for each model type
        
        Args:
            model_type: Type of model ('rf', 'gb', 'xgb', 'lgbm', 'stacking')
        
        Returns:
            Dictionary of parameter ranges
        """
        spaces = {
            'rf': {
                'n_estimators': (50, 500),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': ['balanced', None]
            },
            'gb': {
                'n_estimators': (50, 300),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'subsample': (0.6, 1.0),
                'max_features': ['sqrt', 'log2', None]
            },
            'xgb': {
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_child_weight': (1, 10),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'gamma': (0, 5),
                'reg_alpha': (0, 1),
                'reg_lambda': (1, 2)
            },
            'lgbm': {
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'num_leaves': (10, 300),
                'min_child_samples': (5, 100),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 1),
                'reg_lambda': (0, 1)
            }
        }
        return spaces.get(model_type, {})
    
    def suggest_parameters(self, trial, model_type):
        """
        Suggest parameters for a given model type using Optuna
        
        Args:
            trial: Optuna trial object
            model_type: Type of model
        
        Returns:
            Dictionary of suggested parameters
        """
        space = self.get_model_parameter_space(model_type)
        params = {}
        
        for param_name, param_range in space.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
        
        return params
    
    def create_model(self, model_type, params):
        """
        Create a model instance with given parameters
        
        Args:
            model_type: Type of model
            params: Model parameters
        
        Returns:
            Configured model instance
        """
        if model_type == 'rf':
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        elif model_type == 'gb':
            return GradientBoostingClassifier(random_state=42, **params)
        elif model_type == 'xgb' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
        elif model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1, **params)
        else:
            raise ValueError(f"Unknown or unavailable model type: {model_type}")
    
    def objective_single_model(self, trial, model_type):
        """
        Objective function for single model optimization
        
        Args:
            trial: Optuna trial
            model_type: Type of model to optimize
        
        Returns:
            Cross-validation score to maximize
        """
        try:
            params = self.suggest_parameters(trial, model_type)
            model = self.create_model(model_type, params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.X_train, self.y_train, 
                                   cv=cv, scoring='f1', n_jobs=-1)
            
            return scores.mean()
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            return 0.0
    
    def optimize_single_model(self, model_type, n_trials=None):
        """
        Optimize hyperparameters for a single model
        
        Args:
            model_type: Type of model to optimize
            n_trials: Number of trials (uses self.n_trials if None)
        
        Returns:
            Best parameters and score
        """
        if n_trials is None:
            n_trials = self.n_trials
            
        self.logger.info(f"Starting optimization for {model_type} with {n_trials} trials")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name=f"{model_type}_optimization"
        )
        
        study.optimize(
            lambda trial: self.objective_single_model(trial, model_type),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.logger.info(f"Best {model_type} score: {best_score:.4f}")
        self.logger.info(f"Best {model_type} params: {best_params}")
        
        # Store results
        self.best_params[model_type] = best_params
        self.best_scores[model_type] = best_score
        
        return best_params, best_score
    
    def create_optimized_ensemble(self, base_models_params):
        """
        Create an optimized stacking ensemble from base model parameters
        
        Args:
            base_models_params: Dictionary of optimized parameters for base models
        
        Returns:
            Configured StackingClassifier
        """
        estimators = []
        
        for model_type, params in base_models_params.items():
            if model_type in ['rf', 'gb', 'xgb', 'lgbm']:
                try:
                    model = self.create_model(model_type, params)
                    estimators.append((model_type, model))
                except Exception as e:
                    self.logger.warning(f"Could not create {model_type}: {str(e)}")
        
        if len(estimators) == 0:
            raise ValueError("No valid estimators for ensemble")
        
        # Create stacking classifier with logistic regression meta-learner
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_clf
    
    def optimize_ensemble(self, model_types=None):
        """
        Optimize hyperparameters for ensemble of models
        
        Args:
            model_types: List of model types to include (default: all available)
        
        Returns:
            Optimized ensemble model and results
        """
        if model_types is None:
            model_types = ['rf', 'gb']
            if XGBOOST_AVAILABLE:
                model_types.append('xgb')
            if LIGHTGBM_AVAILABLE:
                model_types.append('lgbm')
        
        self.logger.info(f"Optimizing ensemble with models: {model_types}")
        
        # Optimize each base model
        optimized_params = {}
        for model_type in model_types:
            try:
                best_params, best_score = self.optimize_single_model(model_type)
                optimized_params[model_type] = best_params
            except Exception as e:
                self.logger.error(f"Failed to optimize {model_type}: {str(e)}")
        
        if not optimized_params:
            raise ValueError("No models could be optimized")
        
        # Create and evaluate ensemble
        ensemble = self.create_optimized_ensemble(optimized_params)
        
        # Fit and evaluate ensemble
        ensemble.fit(self.X_train, self.y_train)
        
        # Validation predictions
        y_pred_proba = ensemble.predict_proba(self.X_val)[:, 1]
        y_pred = ensemble.predict(self.X_val)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'f1_score': f1_score(self.y_val, y_pred),
            'auc_score': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        self.logger.info("Ensemble Performance:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        # Find optimal threshold
        optimal_threshold = self.find_optimal_threshold(self.y_val, y_pred_proba)
        
        results = {
            'ensemble': ensemble,
            'optimized_params': optimized_params,
            'metrics': metrics,
            'optimal_threshold': optimal_threshold,
            'feature_names': self.X_train.columns.tolist() if hasattr(self.X_train, 'columns') else None
        }
        
        return results
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """
        Find optimal classification threshold based on F1 score
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            Optimal threshold value
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_thresh)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.logger.info(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        return best_threshold
    
    def save_results(self, results, output_dir="models/hyperparameter_results"):
        """
        Save optimization results to disk
        
        Args:
            results: Results dictionary from optimization
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = output_path / f"optimized_ensemble_{timestamp}.joblib"
        joblib.dump(results, model_path)
        
        # Save parameters as JSON
        params_path = output_path / f"optimization_params_{timestamp}.json"
        params_data = {
            'optimized_params': results['optimized_params'],
            'metrics': results['metrics'],
            'optimal_threshold': results['optimal_threshold'],
            'timestamp': timestamp,
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds
        }
        
        with open(params_path, 'w') as f:
            json.dump(params_data, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        return model_path, params_path

def quick_tune_model(X_train, X_val, y_train, y_val, n_trials=50):
    """
    Quick hyperparameter tuning function for simple usage
    
    Args:
        X_train, X_val, y_train, y_val: Training and validation data
        n_trials: Number of optimization trials
    
    Returns:
        Optimized model and results
    """
    tuner = HyperparameterTuner(X_train, X_val, y_train, y_val, n_trials=n_trials)
    results = tuner.optimize_ensemble()
    return results

if __name__ == "__main__":
    print("HEA Hyperparameter Tuning Module")
    print("This module provides hyperparameter optimization for exoplanet detection models")
    print("Use it in your training scripts or through the webapp interface")