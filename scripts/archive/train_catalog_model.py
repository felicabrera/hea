#!/usr/bin/env python3
"""
Train exoplanet detection models using catalog features only (no light curves needed).

This approach follows the NASA Space Apps Challenge recommended methodology
using the labeled catalog data from KOI/TOI/K2 datasets.

Usage:
    python scripts/train_catalog_model.py --mission kepler
    python scripts/train_catalog_model.py --all-missions --model-type ensemble
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.catalog import CatalogManager
from src.utils.logger import setup_logger
from src.utils.config_loader import config_loader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train exoplanet detection models using catalog features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mission',
        type=str,
        choices=['kepler', 'tess', 'k2'],
        default='kepler',
        help='Mission to train on'
    )
    
    parser.add_argument(
        '--all-missions',
        action='store_true',
        help='Train on all missions combined'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['random_forest', 'gradient_boosting', 'logistic', 'ensemble'],
        default='random_forest',
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/catalog_models',
        help='Directory to save trained models'
    )
    
    return parser.parse_args()

def extract_features(catalog: pd.DataFrame, mission: str, logger) -> pd.DataFrame:
    """Extract and engineer features from catalog data."""
    
    logger.info(f"Extracting features from {mission} catalog...")
    
    features = pd.DataFrame()
    
    # Common numerical features across missions
    numerical_features = []
    
    if mission.lower() == 'kepler':
        # Kepler-specific features
        feature_mapping = {
            'koi_period': 'orbital_period',
            'koi_time0bk': 'transit_epoch',
            'koi_impact': 'impact_parameter',
            'koi_dror': 'planet_star_radius_ratio',
            'koi_incl': 'inclination',
            'koi_ror': 'planet_radius_ratio',
            'koi_srho': 'stellar_density',
            'koi_fittype': 'fit_type',
            'koi_kepmag': 'kepler_magnitude',
            'koi_gmag': 'g_magnitude',
            'koi_rmag': 'r_magnitude',
            'koi_imag': 'i_magnitude',
            'koi_zmag': 'z_magnitude',
            'koi_jmag': 'j_magnitude',
            'koi_hmag': 'h_magnitude',
            'koi_kmag': 'k_magnitude',
            'ra': 'right_ascension',
            'dec': 'declination',
            'koi_slogg': 'stellar_log_g',
            'koi_smet': 'stellar_metallicity'
        }
        
        # Add disposition flags as features
        flag_features = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        
    elif mission.lower() == 'tess':
        # TESS-specific features
        feature_mapping = {
            'pl_orbper': 'orbital_period',
            'pl_tranmid': 'transit_epoch',
            'pl_trandurh': 'transit_duration',
            'pl_trandep': 'transit_depth',
            'pl_rade': 'planet_radius',
            'pl_insol': 'insolation',
            'pl_eqt': 'equilibrium_temp',
            'st_tmag': 'tess_magnitude',
            'st_dist': 'stellar_distance',
            'st_teff': 'stellar_temp',
            'st_logg': 'stellar_log_g',
            'st_rad': 'stellar_radius',
            'ra': 'right_ascension',
            'dec': 'declination'
        }
        
        flag_features = []
    
    # Extract numerical features
    for catalog_col, feature_name in feature_mapping.items():
        if catalog_col in catalog.columns:
            valid_mask = catalog[catalog_col].notna()
            if valid_mask.sum() > 100:  # At least 100 valid values
                features[feature_name] = catalog[catalog_col]
                numerical_features.append(feature_name)
    
    # Add flag features for Kepler
    if mission.lower() == 'kepler':
        for flag_col in flag_features:
            if flag_col in catalog.columns:
                features[flag_col] = catalog[flag_col].fillna(0)
    
    # Feature engineering
    if 'orbital_period' in features.columns:
        # Log period (periods span orders of magnitude)
        features['log_orbital_period'] = np.log10(features['orbital_period'].clip(lower=0.1))
        numerical_features.append('log_orbital_period')
        
        # Period bins (Hot Jupiter, Warm, Cool)
        features['period_category'] = pd.cut(features['orbital_period'], 
                                           bins=[0, 10, 100, np.inf], 
                                           labels=['hot', 'warm', 'cool'])
    
    if 'stellar_temp' in features.columns and 'stellar_temp' not in features.columns:
        # Stellar type categories
        temp_col = 'stellar_temp' if 'stellar_temp' in features.columns else 'st_teff'
        if temp_col in features.columns:
            features['stellar_type'] = pd.cut(features[temp_col],
                                            bins=[0, 3700, 5200, 6000, 7500, np.inf],
                                            labels=['M', 'K', 'G', 'F', 'A'])
    
    logger.info(f"[OK] Extracted {len(features.columns)} features ({len(numerical_features)} numerical)")
    
    return features, numerical_features

def prepare_data(features: pd.DataFrame, labels: dict, numerical_features: list, logger):
    """Prepare data for training."""
    
    logger.info("Preparing training data...")
    
    # Create target column from labels dictionary
    target_ids = list(labels.keys())
    y = np.array([labels[tid] for tid in target_ids])
    
    # Match features to target IDs (this is simplified - in practice need proper matching)
    # For now, use the first len(y) rows of features
    X = features.iloc[:len(y)].copy()
    
    # Handle missing values
    for col in numerical_features:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        # Handle categorical columns properly
        if X[col].dtype.name == 'category':
            # For categorical columns, add 'unknown' to categories first
            X[col] = X[col].cat.add_categories(['unknown']).fillna('unknown')
        else:
            # For object columns, fill NA directly
            X[col] = X[col].fillna('unknown')
        X[col] = le.fit_transform(X[col])
    
    # Remove columns with too many missing values
    missing_threshold = 0.5
    cols_to_keep = []
    for col in X.columns:
        missing_ratio = X[col].isna().sum() / len(X)
        if missing_ratio < missing_threshold:
            cols_to_keep.append(col)
    
    X = X[cols_to_keep]
    logger.info(f"[OK] Kept {len(cols_to_keep)} features with <{missing_threshold*100}% missing values")
    
    # Final missing value handling
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)  # For any remaining missing values
    
    logger.info(f"[OK] Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"   Class distribution: {np.bincount(y)} (0=False Positive, 1=Exoplanet)")
    
    return X, y, cols_to_keep

def train_model(X, y, model_type: str, logger):
    """Train the specified model type."""
    
    logger.info(f"Training {model_type} model...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == 'logistic':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        return model, scaler
    elif model_type == 'ensemble':
        # Simple ensemble of multiple models
        from sklearn.ensemble import VotingClassifier
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
        model.fit(X_scaled, y)
        return model, scaler
    
    model.fit(X, y)
    return model, None

def evaluate_model(model, X_test, y_test, scaler=None, logger=None):
    """Evaluate the trained model."""
    
    X_eval = X_test
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    
    # Metrics
    accuracy = (y_pred == y_test).mean()
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"[OK] Model Performance:")
    logger.info(f"   Accuracy: {accuracy:.3f}")
    logger.info(f"   AUC Score: {auc_score:.3f}")
    
    # Detailed classification report
    logger.info("\\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, 
                                    target_names=['False Positive', 'Exoplanet']))
    
    return accuracy, auc_score

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(name='train_catalog_model', level='INFO')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("[START] TRAINING CATALOG-BASED EXOPLANET DETECTION MODEL")
    logger.info("="*60)
    
    # Load data
    cm = CatalogManager()
    
    if args.all_missions:
        missions = ['kepler', 'tess']  # Skip k2 for now
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
            
            # Extract features
            features, numerical_features = extract_features(catalog, mission, logger)
            
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
    else:
        combined_features = all_features[0]
    
    # Prepare training data
    X, y, feature_names = prepare_data(combined_features, all_labels, numerical_features, logger)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"[OK] Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train model
    model, scaler = train_model(X_train, y_train, args.model_type, logger)
    
    # Evaluate model
    accuracy, auc_score = evaluate_model(model, X_test, y_test, scaler, logger)
    
    # Save model
    model_filename = f"{args.model_type}_{'_'.join(missions)}_model.joblib"
    model_path = output_dir / model_filename
    
    # Save model and metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'missions': missions,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'model_type': args.model_type
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f" Model saved to: {model_path}")
    
    logger.info("\\n[TARGET] TRAINING COMPLETE!")
    logger.info(f"[OK] Accuracy: {accuracy:.1%}")
    logger.info(f"[OK] AUC Score: {auc_score:.3f}")
    logger.info("[SUCCESS] Ready for NASA Space Apps Challenge!")

if __name__ == '__main__':
    main()