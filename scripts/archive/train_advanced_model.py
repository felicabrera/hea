"""
Advanced Exoplanet Detection Model Training with Multiple Algorithms
Includes: Random Forest, KNN, Gradient Boosting, Feature Importance Weighting
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer

from src.data.catalog import CatalogManager

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f'train_advanced_model_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('train_advanced_model')


def extract_all_features(catalog, mission, include_all=True):
    """Extract ALL available features with intelligent handling."""
    
    logger.info(f"Extracting ALL features from {mission} catalog...")
    
    # Define ID columns to exclude per mission
    if mission == 'kepler':
        id_cols = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition']
    elif mission == 'tess':
        id_cols = ['rowid', 'toi', 'toipfx', 'tid', 'ctoi_alias', 'tfopwg_disp']
    elif mission == 'k2':
        id_cols = ['rowid', 'pl_name', 'hostname', 'pl_letter', 'k2_name', 'epic_hostname', 
                   'epic_candname', 'disposition', 'hd_name', 'hip_name']
    else:
        id_cols = ['rowid']
    
    # Get all usable features
    usable_features = []
    feature_coverage = {}
    
    for col in catalog.columns:
        if col in id_cols:
            continue
        
        valid_count = catalog[col].notna().sum()
        coverage = valid_count / len(catalog) * 100
        
        # More lenient threshold for multi-mission training
        threshold = 30 if include_all else 50
        if coverage >= threshold:
            usable_features.append(col)
            feature_coverage[col] = coverage
    
    logger.info(f"Found {len(usable_features)} usable features with ≥{threshold}% coverage")
    
    # Extract features
    features = catalog[usable_features].copy()
    
    # Separate numerical and categorical
    numerical_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()
    
    logger.info(f"[OK] {len(numerical_features)} numerical, {len(categorical_features)} categorical features")
    
    # Add mission identifier as feature
    features['mission_kepler'] = 1 if mission == 'kepler' else 0
    features['mission_tess'] = 1 if mission == 'tess' else 0
    features['mission_k2'] = 1 if mission == 'k2' else 0
    
    # Create engineered features from numerical columns
    engineered_features = pd.DataFrame(index=features.index)
    
    for col in numerical_features:
        if features[col].notna().sum() > len(features) * 0.5:
            # Add log transform for positive values
            if (features[col] > 0).any():
                engineered_features[f'{col}_log'] = np.log1p(features[col].clip(lower=0))
            
            # Add squared term for important features
            if any(key in col.lower() for key in ['period', 'radius', 'temp', 'mass', 'depth', 'duration']):
                engineered_features[f'{col}_squared'] = features[col] ** 2
            
            # Add reciprocal for ratios
            if any(key in col.lower() for key in ['period', 'radius', 'distance']):
                with np.errstate(divide='ignore', invalid='ignore'):
                    recip = 1 / features[col].replace(0, np.nan)
                    engineered_features[f'{col}_reciprocal'] = recip
    
    # Combine all features
    all_features = pd.concat([features, engineered_features], axis=1)
    
    logger.info(f"[OK] Total features after engineering: {len(all_features.columns)}")
    
    return all_features, numerical_features, categorical_features, feature_coverage


def prepare_advanced_data(all_features_list, all_labels_list, feature_selection=True, n_top_features=100):
    """Advanced data preparation with intelligent imputation and selection."""
    
    logger.info("Preparing advanced training data...")
    
    # Combine all mission data
    combined_features = pd.concat(all_features_list, axis=0, ignore_index=True)
    combined_labels = np.concatenate(all_labels_list)
    
    logger.info(f"Combined dataset: {len(combined_features)} samples, {len(combined_features.columns)} features")
    
    # Handle missing values with advanced strategy
    logger.info("Advanced missing value imputation...")
    
    # Separate numerical and categorical
    numerical_cols = combined_features.select_dtypes(include=[np.number]).columns
    categorical_cols = combined_features.select_dtypes(exclude=[np.number]).columns
    
    # Advanced imputation for numerical features
    if len(numerical_cols) > 0:
        # Use median imputer (faster than KNN for large datasets)
        median_imputer = SimpleImputer(strategy='median')
        combined_features[numerical_cols] = median_imputer.fit_transform(combined_features[numerical_cols])
    
    # Simple imputation for categorical
    if len(categorical_cols) > 0:
        simple_imputer = SimpleImputer(strategy='most_frequent')
        combined_features[categorical_cols] = simple_imputer.fit_transform(combined_features[categorical_cols])
    
    # Convert categorical to numerical
    for col in categorical_cols:
        combined_features[col] = pd.factorize(combined_features[col])[0]
    
    # Remove constant features
    constant_features = [col for col in combined_features.columns 
                        if combined_features[col].nunique() <= 1]
    if constant_features:
        logger.info(f"Removing {len(constant_features)} constant features")
        combined_features = combined_features.drop(columns=constant_features)
    
    # Remove highly correlated features
    logger.info("Removing highly correlated features...")
    corr_matrix = combined_features.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [col for col in upper_triangle.columns 
               if any(upper_triangle[col] > 0.95)]
    
    if to_drop:
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        combined_features = combined_features.drop(columns=to_drop)
    
    # Feature selection using F-statistic (faster than mutual information)
    if feature_selection and len(combined_features.columns) > n_top_features:
        logger.info(f"Selecting top {n_top_features} features using F-statistic...")
        
        selector = SelectKBest(score_func=f_classif, k=min(n_top_features, len(combined_features.columns)))
        X_selected = selector.fit_transform(combined_features, combined_labels)
        
        selected_features = combined_features.columns[selector.get_support()].tolist()
        combined_features = pd.DataFrame(X_selected, columns=selected_features)
        
        # Store feature scores for importance weighting
        feature_scores = dict(zip(combined_features.columns, selector.scores_[selector.get_support()]))
        logger.info(f"[OK] Selected {len(selected_features)} most informative features")
    else:
        feature_scores = None
    
    logger.info(f"[OK] Final dataset: {len(combined_features)} samples, {len(combined_features.columns)} features")
    logger.info(f"   Class distribution: {np.bincount(combined_labels)} (0=False Positive, 1=Exoplanet)")
    
    return combined_features, combined_labels, feature_scores


def calculate_sample_weights(y, feature_scores=None):
    """Calculate sample weights based on class imbalance and feature importance."""
    
    from sklearn.utils.class_weight import compute_class_weight
    
    # Base weights for class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    sample_weights = np.array([class_weights[label] for label in y])
    
    logger.info(f"Class weights: {dict(enumerate(class_weights))}")
    
    return sample_weights


def train_advanced_ensemble(X_train, y_train, X_test, y_test, sample_weights, optimize=False):
    """Train advanced ensemble with Random Forest, KNN, and Gradient Boosting."""
    
    logger.info("Training ADVANCED ENSEMBLE with multiple algorithms...")
    
    # Split sample weights
    train_weights = sample_weights[:len(X_train)]
    
    # 1. Random Forest with optimization
    logger.info("Training Random Forest...")
    if optimize:
        rf_params = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }
        rf = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            rf_params,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        rf.fit(X_train, y_train, sample_weight=train_weights)
        rf_model = rf.best_estimator_
        logger.info(f"Best RF params: {rf.best_params_}")
    else:
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train, sample_weight=train_weights)
    
    # 2. Gradient Boosting with optimization
    logger.info("Training Gradient Boosting...")
    if optimize:
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0]
        }
        gb = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            gb_params,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        gb.fit(X_train, y_train, sample_weight=train_weights)
        gb_model = gb.best_estimator_
        logger.info(f"Best GB params: {gb.best_params_}")
    else:
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=2,
            subsample=0.9,
            random_state=42
        )
        gb_model.fit(X_train, y_train, sample_weight=train_weights)
    
    # 3. K-Nearest Neighbors with optimization
    logger.info("Training K-Nearest Neighbors...")
    
    # Scale features for KNN (distance-based algorithm)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if optimize:
        knn_params = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'p': [1, 2]
        }
        knn = GridSearchCV(
            KNeighborsClassifier(n_jobs=-1),
            knn_params,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        knn.fit(X_train_scaled, y_train)
        knn_model = knn.best_estimator_
        logger.info(f"Best KNN params: {knn.best_params_}")
    else:
        knn_model = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        )
        knn_model.fit(X_train_scaled, y_train)
    
    # 4. Create weighted voting ensemble
    logger.info("Creating weighted voting ensemble...")
    
    # Get individual model scores for weighting
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    knn_pred = knn_model.predict(X_test_scaled)
    
    rf_f1 = f1_score(y_test, rf_pred)
    gb_f1 = f1_score(y_test, gb_pred)
    knn_f1 = f1_score(y_test, knn_pred)
    
    logger.info(f"Individual F1 scores - RF: {rf_f1:.3f}, GB: {gb_f1:.3f}, KNN: {knn_f1:.3f}")
    
    # Weighted ensemble prediction
    total_f1 = rf_f1 + gb_f1 + knn_f1
    rf_weight = rf_f1 / total_f1
    gb_weight = gb_f1 / total_f1
    knn_weight = knn_f1 / total_f1
    
    logger.info(f"Ensemble weights - RF: {rf_weight:.3f}, GB: {gb_weight:.3f}, KNN: {knn_weight:.3f}")
    
    # Weighted prediction
    ensemble_pred_proba = (
        rf_weight * rf_model.predict_proba(X_test)[:, 1] +
        gb_weight * gb_model.predict_proba(X_test)[:, 1] +
        knn_weight * knn_model.predict_proba(X_test_scaled)[:, 1]
    )
    ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
    
    # Evaluate ensemble
    accuracy = accuracy_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred)
    recall = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_pred_proba)
    
    logger.info("[OK] ADVANCED ENSEMBLE PERFORMANCE:")
    logger.info(f"   Accuracy: {accuracy:.3f}")
    logger.info(f"   AUC Score: {auc:.3f}")
    logger.info(f"   Precision: {precision:.3f}")
    logger.info(f"   Recall: {recall:.3f}")
    logger.info(f"   F1 Score: {f1:.3f}")
    
    # Confusion matrix details
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, ensemble_pred).ravel()
    logger.info(f"   True Positives: {tp}")
    logger.info(f"   False Positives: {fp}")
    logger.info(f"   False Negatives: {fn}")
    logger.info(f"   True Negatives: {tn}")
    
    # Feature importance from RF and GB
    logger.info("\n[TARGET] TOP 15 MOST IMPORTANT FEATURES:")
    rf_importance = pd.DataFrame({
        'feature': X_train.columns if hasattr(X_train, 'columns') else [f'f_{i}' for i in range(X_train.shape[1])],
        'rf_importance': rf_model.feature_importances_,
        'gb_importance': gb_model.feature_importances_
    })
    rf_importance['avg_importance'] = (rf_importance['rf_importance'] + rf_importance['gb_importance']) / 2
    rf_importance = rf_importance.sort_values('avg_importance', ascending=False)
    
    for idx, row in rf_importance.head(15).iterrows():
        logger.info(f"   {row['feature']}: {row['avg_importance']:.4f} (RF: {row['rf_importance']:.4f}, GB: {row['gb_importance']:.4f})")
    
    # Store models
    models = {
        'rf': rf_model,
        'gb': gb_model,
        'knn': knn_model,
        'scaler': scaler,
        'weights': {'rf': rf_weight, 'gb': gb_weight, 'knn': knn_weight},
        'feature_importance': rf_importance
    }
    
    return models, accuracy, auc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Train advanced exoplanet detection model')
    parser.add_argument('--mission', type=str, default='kepler',
                       choices=['kepler', 'tess', 'k2'],
                       help='Mission to train on')
    parser.add_argument('--all-missions', action='store_true',
                       help='Train on all missions combined')
    parser.add_argument('--optimize', action='store_true',
                       help='Perform hyperparameter optimization (slower)')
    parser.add_argument('--n-features', type=int, default=100,
                       help='Number of top features to select')
    parser.add_argument('--no-feature-selection', action='store_true',
                       help='Disable feature selection')
    
    args = parser.parse_args()
    
    logger.info("[START] TRAINING ADVANCED EXOPLANET DETECTION MODEL")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  - Algorithms: Random Forest + Gradient Boosting + KNN")
    logger.info(f"  - Feature Selection: {not args.no_feature_selection}")
    logger.info(f"  - Optimization: {args.optimize}")
    logger.info(f"  - Sample Weighting: Enabled")
    
    # Load data
    cm = CatalogManager()
    
    if args.all_missions:
        missions = ['kepler', 'tess', 'k2']
        logger.info(f"Training on ALL missions: {missions}")
    else:
        missions = [args.mission]
        logger.info(f"Training on {args.mission} mission")
    
    all_features_list = []
    all_labels_list = []
    
    for mission in missions:
        try:
            catalog = cm.load_catalog(mission)
            labels = cm.get_labels(mission)
            
            # Extract features
            features, num_feats, cat_feats, coverage = extract_all_features(
                catalog, mission, include_all=args.all_missions
            )
            
            # Create target IDs based on mission
            if mission == 'kepler':
                target_ids = [f"KIC {int(row['kepid'])}" for _, row in catalog.iterrows()]
            elif mission == 'tess':
                target_ids = [f"TIC {int(row['tid'])}" for _, row in catalog.iterrows()]
            elif mission == 'k2':
                target_ids = [row['epic_hostname'] if isinstance(row['epic_hostname'], str) else '' 
                             for _, row in catalog.iterrows()]
            
            features['target_id'] = target_ids
            labels_df = pd.DataFrame(list(labels.items()), columns=['target_id', 'label'])
            
            merged = features.merge(labels_df, on='target_id', how='inner')
            
            feature_cols = [c for c in merged.columns if c not in ['target_id', 'label']]
            X = merged[feature_cols]
            y = merged['label'].values
            
            logger.info(f"[OK] {mission.upper()}: {len(X)} samples with labels")
            
            all_features_list.append(X)
            all_labels_list.append(y)
            
        except Exception as e:
            logger.error(f"[FAIL] Error processing {mission}: {str(e)}")
            continue
    
    if not all_features_list:
        logger.error("No data loaded! Exiting.")
        return
    
    # Prepare data
    X, y, feature_scores = prepare_advanced_data(
        all_features_list,
        all_labels_list,
        feature_selection=not args.no_feature_selection,
        n_top_features=args.n_features
    )
    
    # Calculate sample weights
    sample_weights = calculate_sample_weights(y, feature_scores)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_weights = sample_weights[:len(X_train)]
    
    logger.info(f"[OK] Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train advanced ensemble
    models, accuracy, auc, precision, recall, f1 = train_advanced_ensemble(
        X_train, y_train, X_test, y_test, sample_weights, optimize=args.optimize
    )
    
    # Save models
    model_dir = Path('models/advanced_models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    mission_str = '_'.join(missions) if args.all_missions else args.mission
    model_path = model_dir / f'advanced_ensemble_{mission_str}_model.joblib'
    
    joblib.dump(models, model_path)
    logger.info(f" Model saved to: {model_path}")
    
    logger.info("\n[SUCCESS] ADVANCED TRAINING COMPLETE!")
    logger.info(f"[OK] Accuracy: {accuracy*100:.1f}%")
    logger.info(f"[OK] AUC Score: {auc:.3f}")
    logger.info(f"[OK] Precision: {precision:.3f}")
    logger.info(f"[OK] Recall: {recall:.3f}")
    logger.info(f"[OK] F1 Score: {f1:.3f}")
    logger.info("[START] NASA Space Apps Challenge Ready!")


if __name__ == '__main__':
    main()
