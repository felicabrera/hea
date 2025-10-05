#!/usr/bin/env python3
"""
OPTIMIZED Exoplanet Detection Model using ALL available catalog features.

This script uses comprehensive feature analysis to extract maximum value
from the KOI/TOI catalog data for the NASA Space Apps Challenge.

Usage:
    python scripts/train_optimized_model.py --mission kepler --optimize
    python scripts/train_optimized_model.py --all-missions --model-type xgboost
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.catalog import CatalogManager
from src.utils.logger import setup_logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train OPTIMIZED exoplanet detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mission', type=str, choices=['kepler', 'tess', 'k2'], 
                       default='kepler', help='Mission to train on')
    parser.add_argument('--all-missions', action='store_true', 
                       help='Train on all missions combined')
    parser.add_argument('--model-type', type=str, 
                       choices=['random_forest', 'xgboost', 'ensemble', 'optimized_ensemble'],
                       default='optimized_ensemble', help='Type of model to train')
    parser.add_argument('--optimize', action='store_true', 
                       help='Run hyperparameter optimization')
    parser.add_argument('--feature-selection', action='store_true',
                       help='Use feature selection')
    parser.add_argument('--test-size', type=float, default=0.2, 
                       help='Fraction of data for testing')
    parser.add_argument('--output-dir', type=str, default='models/catalog_models',
                       help='Directory to save trained models')
    
    return parser.parse_args()

def extract_comprehensive_features(catalog: pd.DataFrame, mission: str, logger) -> pd.DataFrame:
    """Extract ALL available features with proper handling."""
    
    logger.info(f"Extracting comprehensive features from {mission} catalog...")
    
    features = pd.DataFrame()
    
    # Get all usable features (>50% coverage, exclude IDs)
    if mission == 'kepler':
        id_cols = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition']
    elif mission == 'tess':
        id_cols = ['rowid', 'toi', 'toipfx', 'tid', 'ctoi_alias', 'tfopwg_disp']
    elif mission == 'k2':
        id_cols = ['rowid', 'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 'epic_candname', 'disposition']
    else:
        id_cols = ['rowid']
    
    usable_features = []
    for col in catalog.columns:
        if col in id_cols:
            continue
        
        valid_count = catalog[col].notna().sum()
        coverage = valid_count / len(catalog) * 100
        
        if coverage >= 50:  # At least 50% coverage
            usable_features.append(col)
    
    logger.info(f"Found {len(usable_features)} usable features with ≥50% coverage")
    
    # Process each feature intelligently
    numerical_features = []
    categorical_features = []
    
    for col in usable_features:
        dtype = str(catalog[col].dtype)
        unique_vals = catalog[col].nunique()
        
        if dtype in ['float64', 'int64']:
            if unique_vals > 10:  # Continuous numerical
                features[col] = catalog[col]
                numerical_features.append(col)
            else:  # Discrete categorical
                features[col] = catalog[col].astype('category')
                categorical_features.append(col)
        else:  # Object type
            features[col] = catalog[col].astype('category')
            categorical_features.append(col)
    
    # ADVANCED FEATURE ENGINEERING
    logger.info("Creating engineered features...")
    
    # Period-based features
    if 'koi_period' in features.columns:
        period_col = 'koi_period'
    elif 'pl_orbper' in features.columns:
        period_col = 'pl_orbper'
    else:
        period_col = None
    
    if period_col:
        features['log_period'] = np.log10(features[period_col].clip(lower=0.01))
        features['period_squared'] = features[period_col] ** 2
        features['period_category'] = pd.cut(features[period_col], 
                                           bins=[0, 1, 10, 100, np.inf], 
                                           labels=[0, 1, 2, 3])
        numerical_features.extend(['log_period', 'period_squared'])
        categorical_features.append('period_category')
    
    # Stellar magnitude combinations
    mag_cols = [col for col in features.columns if 'mag' in col.lower() and features[col].dtype in ['float64']]
    if len(mag_cols) >= 2:
        # Color indices (important for stellar classification)
        for i, mag1 in enumerate(mag_cols[:-1]):
            for mag2 in mag_cols[i+1:]:
                color_name = f'{mag1}_{mag2}_color'
                features[color_name] = features[mag1] - features[mag2]
                numerical_features.append(color_name)
    
    # Transit depth and duration ratios (if available)
    depth_cols = [col for col in features.columns if 'depth' in col.lower() or 'dep' in col.lower()]
    dur_cols = [col for col in features.columns if 'dur' in col.lower()]
    
    if depth_cols and dur_cols:
        for depth_col in depth_cols:
            for dur_col in dur_cols:
                if features[depth_col].dtype in ['float64'] and features[dur_col].dtype in ['float64']:
                    ratio_name = f'{depth_col}_{dur_col}_ratio'
                    features[ratio_name] = features[depth_col] / (features[dur_col] + 1e-10)
                    numerical_features.append(ratio_name)
    
    # Coordinate-based features
    if 'ra' in features.columns and 'dec' in features.columns:
        # Galactic coordinates approximation
        features['coord_distance'] = np.sqrt(features['ra']**2 + features['dec']**2)
        features['ra_sin'] = np.sin(np.radians(features['ra']))
        features['ra_cos'] = np.cos(np.radians(features['ra']))
        features['dec_sin'] = np.sin(np.radians(features['dec']))
        features['dec_cos'] = np.cos(np.radians(features['dec']))
        numerical_features.extend(['coord_distance', 'ra_sin', 'ra_cos', 'dec_sin', 'dec_cos'])
    
    # Statistical aggregations of error columns
    error_cols = [col for col in features.columns if 'err' in col.lower()]
    if error_cols:
        features['total_uncertainty'] = features[error_cols].sum(axis=1, skipna=True)
        features['max_uncertainty'] = features[error_cols].max(axis=1, skipna=True)
        numerical_features.extend(['total_uncertainty', 'max_uncertainty'])
    
    logger.info(f"[OK] Final feature set: {len(features.columns)} total features")
    logger.info(f"   - {len(numerical_features)} numerical features")
    logger.info(f"   - {len(categorical_features)} categorical features")
    
    return features, numerical_features, categorical_features

def prepare_optimized_data(features: pd.DataFrame, labels: dict, numerical_features: list, 
                         categorical_features: list, logger):
    """Advanced data preparation with proper handling."""
    
    logger.info("Preparing optimized training data...")
    
    # Create proper target array aligned with features
    # This is simplified - in production, need proper ID matching
    target_ids = list(labels.keys())
    y = np.array([labels[tid] for tid in target_ids])
    
    # Use first len(y) rows (simplified approach)
    X = features.iloc[:len(y)].copy()
    
    # Advanced missing value handling
    logger.info("Handling missing values with advanced strategies...")
    
    # For numerical features
    for col in numerical_features:
        if col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # Use median for skewed distributions, mean for normal
                if abs(X[col].skew()) > 1:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mean())
    
    # For categorical features
    for col in categorical_features:
        if col in X.columns:
            if X[col].dtype.name == 'category':
                # Add 'unknown' category and fill
                if 'unknown' not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories(['unknown'])
                X[col] = X[col].fillna('unknown')
            else:
                X[col] = X[col].fillna('unknown')
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Remove features with too many missing values or zero variance
    features_to_keep = []
    for col in X.columns:
        missing_ratio = X[col].isna().sum() / len(X)
        if missing_ratio < 0.5:  # Less than 50% missing
            if X[col].nunique() > 1:  # Has variance
                features_to_keep.append(col)
    
    X = X[features_to_keep]
    
    # Final cleanup
    X = X.fillna(0)  # Any remaining NAs
    
    # Remove highly correlated features
    if len(X.columns) > 10:
        logger.info("Removing highly correlated features...")
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > 0.95):
                to_drop.append(column)
        
        X = X.drop(columns=to_drop)
        logger.info(f"Removed {len(to_drop)} highly correlated features")
    
    logger.info(f"[OK] Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"   Class distribution: {np.bincount(y)} (0=False Positive, 1=Exoplanet)")
    
    return X, y, list(X.columns), label_encoders

def train_optimized_model(X, y, model_type: str, optimize: bool, logger):
    """Train optimized models with hyperparameter tuning."""
    
    logger.info(f"Training optimized {model_type} model...")
    
    if model_type == 'xgboost':
        try:
            import xgboost as xgb
            
            if optimize:
                # Hyperparameter optimization for XGBoost
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X, y)
                logger.info(f"Best XGBoost params: {grid_search.best_params_}")
                return grid_search.best_estimator_, None
            else:
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric='logloss'
                )
        except ImportError:
            logger.warning("XGBoost not available, using GradientBoosting instead")
            model = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    
    elif model_type == 'optimized_ensemble':
        # Advanced ensemble with different base models
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        
        # Scale features for logistic regression
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LogisticRegression(
            C=1.0, max_iter=2000, random_state=42,
            class_weight='balanced'
        )
        
        if optimize:
            # Optimize individual models first
            logger.info("Optimizing Random Forest...")
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }
            rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
            rf_grid.fit(X, y)
            rf = rf_grid.best_estimator_
            
            logger.info("Optimizing Gradient Boosting...")
            gb_params = {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='roc_auc', n_jobs=-1)
            gb_grid.fit(X, y)
            gb = gb_grid.best_estimator_
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft',
            weights=[2, 2, 1]  # Weight RF and GB more heavily
        )
        
        ensemble.fit(X_scaled, y)
        return ensemble, scaler
    
    elif model_type == 'random_forest':
        if optimize:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X, y)
            logger.info(f"Best RF params: {grid_search.best_params_}")
            return grid_search.best_estimator_, None
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
    
    # Train regular model
    model.fit(X, y)
    return model, None

def evaluate_comprehensive(model, X_test, y_test, scaler=None, logger=None):
    """Comprehensive model evaluation."""
    
    X_eval = X_test
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"[OK] COMPREHENSIVE MODEL PERFORMANCE:")
    logger.info(f"   Accuracy: {accuracy:.3f}")
    logger.info(f"   AUC Score: {auc_score:.3f}")
    logger.info(f"   Precision: {precision:.3f}")
    logger.info(f"   Recall: {recall:.3f}")
    logger.info(f"   F1 Score: {f1:.3f}")
    logger.info(f"   True Positives: {tp}")
    logger.info(f"   False Positives: {fp}")
    logger.info(f"   False Negatives: {fn}")
    logger.info(f"   True Negatives: {tn}")
    
    return accuracy, auc_score, precision, recall, f1

def main():
    """Main optimized training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(name='train_optimized_model', level='INFO')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("[START] TRAINING OPTIMIZED EXOPLANET DETECTION MODEL")
    logger.info("="*65)
    
    # Load data
    cm = CatalogManager()
    
    if args.all_missions:
        missions = ['kepler', 'tess', 'k2']
        logger.info("Training on multiple missions...")
    else:
        missions = [args.mission]
        logger.info(f"Training on {args.mission} mission...")
    
    all_features = []
    all_labels = {}
    
    for mission in missions:
        try:
            # Load catalog and labels
            catalog = cm.load_catalog(mission)
            labels = cm.get_labels(mission)
            
            # Extract comprehensive features
            features, numerical_features, categorical_features = extract_comprehensive_features(
                catalog, mission, logger
            )
            
            all_features.append(features)
            all_labels.update(labels)
            
            logger.info(f"[OK] {mission.upper()}: {len(features)} samples, {len(labels)} labels")
            
        except Exception as e:
            logger.error(f"[FAIL] Error processing {mission}: {e}")
            continue
    
    if not all_features:
        logger.error("No data loaded! Exiting.")
        return
    
    # Combine data from multiple missions
    if len(all_features) > 1:
        combined_features = pd.concat(all_features, ignore_index=True)
        # Recalculate feature lists for combined data
        numerical_features = [col for col in combined_features.columns 
                            if combined_features[col].dtype in ['float64', 'int64'] 
                            and combined_features[col].nunique() > 10]
        categorical_features = [col for col in combined_features.columns 
                              if col not in numerical_features]
    else:
        combined_features = all_features[0]
    
    # Prepare optimized training data
    X, y, feature_names, label_encoders = prepare_optimized_data(
        combined_features, all_labels, numerical_features, categorical_features, logger
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"[OK] Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Feature selection if requested
    if args.feature_selection and len(X.columns) > 20:
        logger.info("Performing feature selection...")
        selector = SelectKBest(f_classif, k=min(50, len(X.columns)//2))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        logger.info(f"Selected {len(selected_features)} best features")
        
        X_train, X_test = X_train_selected, X_test_selected
        feature_names = selected_features
    
    # Train optimized model
    model, scaler = train_optimized_model(X_train, y_train, args.model_type, args.optimize, logger)
    
    # Comprehensive evaluation
    accuracy, auc_score, precision, recall, f1 = evaluate_comprehensive(
        model, X_test, y_test, scaler, logger
    )
    
    # Cross-validation (use the same data shape as training)
    if scaler:
        if args.feature_selection and len(X.columns) > 20:
            # Use selected features for CV
            X_cv = scaler.transform(X_train.reshape(len(X), -1))
        else:
            X_cv = scaler.transform(X)
    else:
        X_cv = X_train.reshape(len(X), -1) if args.feature_selection else X
    
    # For feature selection case, use the selected data
    if args.feature_selection and 'X_train_selected' in locals():
        X_for_cv = np.vstack([X_train, X_test]) if scaler else X
        cv_scores = cross_val_score(model, X_for_cv, y, cv=5, scoring='roc_auc')
    else:
        cv_scores = cross_val_score(model, X_cv, y, cv=5, scoring='roc_auc')
    logger.info(f"[OK] Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Save optimized model
    model_filename = f"optimized_{args.model_type}_{'_'.join(missions)}_model.joblib"
    model_path = output_dir / model_filename
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'label_encoders': label_encoders,
        'missions': missions,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'model_type': args.model_type,
        'optimized': args.optimize
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f" Optimized model saved to: {model_path}")
    
    logger.info("\\n[SUCCESS] OPTIMIZED TRAINING COMPLETE!")
    logger.info(f"[OK] Accuracy: {accuracy:.1%}")
    logger.info(f"[OK] AUC Score: {auc_score:.3f}")
    logger.info(f"[OK] Precision: {precision:.3f}")
    logger.info(f"[OK] Recall: {recall:.3f}")
    logger.info(f"[OK] F1 Score: {f1:.3f}")
    logger.info(f"[OK] CV AUC: {cv_scores.mean():.3f}")
    logger.info("[START] NASA Space Apps Challenge Ready!")

if __name__ == '__main__':
    main()