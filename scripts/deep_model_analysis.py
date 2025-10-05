"""
Deep Model Analysis Script
Comprehensive investigation to identify accuracy improvement opportunities
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.catalog import CatalogManager

def analyze_current_model():
    """Analyze the current best model in detail."""
    
    print("=" * 80)
    print("DEEP MODEL ANALYSIS - Accuracy Improvement Investigation")
    print("=" * 80)
    
    # Load model
    model_path = Path('models/catalog_models/optimized_optimized_ensemble_kepler_tess_k2_model.joblib')
    model_data = joblib.load(model_path)
    
    print("\n1. CURRENT MODEL ARCHITECTURE")
    print("-" * 80)
    print(f"Model Type: {model_data.get('model_type', 'Unknown')}")
    print(f"Base Model: {type(model_data['model']).__name__}")
    
    if hasattr(model_data['model'], 'estimators_'):
        print(f"\nEnsemble Estimators ({len(model_data['model'].estimators_)}):")
        for est in model_data['model'].estimators_:
            print(f"  - {type(est).__name__}")
            if hasattr(est, 'n_estimators'):
                print(f"      n_estimators: {est.n_estimators}")
            if hasattr(est, 'max_depth'):
                print(f"      max_depth: {est.max_depth}")
        print(f"\nVoting Strategy: {model_data['model'].voting}")
        if hasattr(model_data['model'], 'weights'):
            print(f"Weights: {model_data['model'].weights}")
    
    print(f"\n2. FEATURE ANALYSIS")
    print("-" * 80)
    print(f"Total Features Used: {len(model_data['feature_names'])}")
    print(f"\nFirst 15 features:")
    for i, f in enumerate(model_data['feature_names'][:15], 1):
        print(f"  {i:2d}. {f}")
    print(f"\n  ... ({len(model_data['feature_names']) - 20} features omitted) ...")
    print(f"\nLast 5 features:")
    for i, f in enumerate(model_data['feature_names'][-5:], len(model_data['feature_names'])-4):
        print(f"  {i:2d}. {f}")
    
    print(f"\n3. TRAINING DATA ANALYSIS")
    print("-" * 80)
    
    # Analyze class distribution per mission
    cat = CatalogManager()
    
    missions = ['kepler', 'tess', 'k2']
    total_samples = 0
    total_pos = 0
    total_neg = 0
    
    for mission in missions:
        labels = cat.get_labels(mission)
        labels_array = np.array(list(labels.values()))
        n_pos = np.sum(labels_array == 1)
        n_neg = np.sum(labels_array == 0)
        imbalance = n_pos / n_neg if n_neg > 0 else 0
        
        total_samples += len(labels_array)
        total_pos += n_pos
        total_neg += n_neg
        
        print(f"\n{mission.upper()}:")
        print(f"  Total: {len(labels_array):,} samples")
        print(f"  Positives (Exoplanets): {n_pos:,} ({n_pos/len(labels_array)*100:.1f}%)")
        print(f"  Negatives (False Pos):  {n_neg:,} ({n_neg/len(labels_array)*100:.1f}%)")
        print(f"  Imbalance Ratio: {imbalance:.2f}:1")
    
    print(f"\nCOMBINED DATASET:")
    print(f"  Total: {total_samples:,} samples")
    print(f"  Positives: {total_pos:,} ({total_pos/total_samples*100:.1f}%)")
    print(f"  Negatives: {total_neg:,} ({total_neg/total_samples*100:.1f}%)")
    print(f"  Overall Imbalance: {total_pos/total_neg:.2f}:1")
    
    print(f"\n4. PERFORMANCE BREAKDOWN")
    print("-" * 80)
    print(f"Accuracy:  {model_data['accuracy']:.1%}")
    print(f"Precision: {model_data['precision']:.1%}")
    print(f"Recall:    {model_data['recall']:.1%}")
    print(f"F1 Score:  {model_data['f1_score']:.3f}")
    print(f"AUC:       {model_data['auc_score']:.3f}")
    print(f"CV AUC:    {model_data.get('cv_auc_mean', 0):.3f} ± {model_data.get('cv_auc_std', 0):.3f}")
    
    # Calculate what could be improved
    print(f"\n5. ERROR ANALYSIS")
    print("-" * 80)
    print(f"Current Error Rate: {(1 - model_data['accuracy'])*100:.1f}%")
    print(f"  → {(1 - model_data['accuracy'])*3437:.0f} misclassified samples out of 3,437")
    print(f"\nRecall Gap: {(1 - model_data['recall'])*100:.1f}%")
    print(f"  → Missing {(1 - model_data['recall'])*100:.1f}% of actual exoplanets")
    print(f"\nPrecision Gap: {(1 - model_data['precision'])*100:.1f}%")
    print(f"  → {(1 - model_data['precision'])*100:.1f}% false alarm rate")
    
    return model_data

def identify_improvement_opportunities(model_data):
    """Identify specific opportunities to improve accuracy."""
    
    print(f"\n{'=' * 80}")
    print("IMPROVEMENT OPPORTUNITIES - ACTIONABLE RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n[TARGET] GOAL: Increase accuracy from 69.5% to 75%+ (Need +5.5 percentage points)")
    
    print(f"\n{'=' * 80}")
    print("STRATEGY 1: ADVANCED ENSEMBLE METHODS (+2-3%)")
    print("=" * 80)
    
    print(f"""
Current: VotingClassifier with RF + GB + LR (soft voting, static weights)

IMPROVEMENTS:
[OK] 1. Add XGBoost to ensemble
   - XGBoost often outperforms GB on imbalanced data
   - Expected gain: +1-2% accuracy
   - Action: Install xgboost, add XGBClassifier with optimized params

[OK] 2. Implement Stacking Ensemble
   - Use RF, GB, XGB as base models
   - Use LR or GB as meta-learner
   - Expected gain: +0.5-1.5% accuracy
   - Action: Use StackingClassifier instead of VotingClassifier

[OK] 3. Dynamic Weight Optimization
   - Current: Static weights [2, 2, 1]
   - Optimize weights using validation set performance
   - Expected gain: +0.3-0.5% accuracy
   - Action: Use GridSearch for optimal ensemble weights

[OK] 4. Add LightGBM
   - Faster training, often better than standard GB
   - Handles categorical features natively
   - Expected gain: +0.5-1% accuracy
   - Action: Install lightgbm, add to ensemble
""")
    
    print(f"\n{'=' * 80}")
    print("STRATEGY 2: ADVANCED FEATURE ENGINEERING (+1-2%)")
    print("=" * 80)
    
    print(f"""
Current: {len(model_data['feature_names'])} features (basic engineering)

IMPROVEMENTS:
[OK] 5. Domain-Specific Astronomical Features
   - Transit depth ratio: (planet_radius / star_radius)²
   - Orbital velocity: 2π × semi_major_axis / period
   - Equilibrium temperature calculation
   - Signal-to-noise ratio features
   - Expected gain: +0.5-1% accuracy

[OK] 6. Interaction Features
   - period × planet_radius (size-orbit relationship)
   - stellar_mass × orbital_period (Kepler's law)
   - transit_duration × period (geometry)
   - Expected gain: +0.3-0.7% accuracy

[OK] 7. Polynomial Features (degree=2)
   - Create squared and interaction terms for top 10 features
   - Use SelectKBest to keep only informative ones
   - Expected gain: +0.2-0.5% accuracy

[OK] 8. Time-Series Features (if light curve data available)
   - Fourier transforms of flux signals
   - Wavelet features
   - Periodicity detection
   - Expected gain: +1-2% accuracy if light curves used
""")
    
    print(f"\n{'=' * 80}")
    print("STRATEGY 3: HANDLE CLASS IMBALANCE BETTER (+1-2%)")
    print("=" * 80)
    
    print(f"""
Current: Class weights applied, but TESS/K2 highly imbalanced (4.77:1, 5.24:1)

IMPROVEMENTS:
[OK] 9. SMOTE (Synthetic Minority Oversampling)
   - Generate synthetic samples for minority class
   - Use SMOTE or ADASYN
   - Expected gain: +0.5-1.5% accuracy
   - Action: from imblearn.over_sampling import SMOTE

[OK] 10. Advanced Sampling Strategies
   - SMOTETomek: SMOTE + Tomek links cleaning
   - BorderlineSMOTE: Focus on borderline cases
   - Expected gain: +0.5-1% accuracy

[OK] 11. Threshold Optimization
   - Current: Using 0.5 probability threshold
   - Optimize threshold for best F1 or accuracy
   - Expected gain: +0.3-0.8% accuracy

[OK] 12. Cost-Sensitive Learning
   - Assign different misclassification costs
   - Penalize false negatives more heavily
   - Expected gain: +0.2-0.5% recall/accuracy
""")
    
    print(f"\n{'=' * 80}")
    print("STRATEGY 4: HYPERPARAMETER OPTIMIZATION (+0.5-1.5%)")
    print("=" * 80)
    
    print(f"""
Current: Manual hyperparameters, limited GridSearch

IMPROVEMENTS:
[OK] 13. Bayesian Optimization
   - Use Optuna or Hyperopt
   - More efficient than GridSearch
   - Expected gain: +0.3-1% accuracy
   - Action: pip install optuna

[OK] 14. Broader Parameter Search
   - RF: Try n_estimators up to 1000
   - GB: Try learning rates 0.001 to 0.3
   - Max depth: Try up to 30
   - Expected gain: +0.2-0.5% accuracy

[OK] 15. Feature Selection Optimization
   - Current: Using top 80 features
   - Try 50, 100, 150 features
   - Use RFE (Recursive Feature Elimination)
   - Expected gain: +0.3-0.7% accuracy
""")
    
    print(f"\n{'=' * 80}")
    print("STRATEGY 5: DEEP LEARNING APPROACHES (+2-4%)")
    print("=" * 80)
    
    print(f"""
Current: Traditional ML only (RF, GB, LR)

IMPROVEMENTS:
[OK] 16. Neural Network Classifier
   - MLP with 2-3 hidden layers
   - Dropout for regularization
   - Expected gain: +0.5-1.5% accuracy
   - Action: Use MLPClassifier or TensorFlow

[OK] 17. Attention-Based Models
   - If using light curves: 1D CNN + Attention
   - Transformer for time-series
   - Expected gain: +1-3% accuracy (requires light curves)

[OK] 18. Ensemble NN + Traditional ML
   - Train separate neural network
   - Combine with RF/GB using stacking
   - Expected gain: +0.5-1% accuracy
""")
    
    print(f"\n{'=' * 80}")
    print("STRATEGY 6: DATA QUALITY & AUGMENTATION (+1-2%)")
    print("=" * 80)
    
    print(f"""
Current: Using catalog features only, median imputation

IMPROVEMENTS:
[OK] 19. Advanced Imputation
   - Use IterativeImputer (MICE algorithm)
   - KNN imputation for similar stars
   - Expected gain: +0.2-0.5% accuracy

[OK] 20. Feature Scaling Optimization
   - Try different scalers: RobustScaler, PowerTransformer
   - Log/Box-Cox transforms for skewed features
   - Expected gain: +0.2-0.4% accuracy

[OK] 21. Outlier Detection & Handling
   - Identify and handle outliers in features
   - Use IsolationForest or LOF
   - Expected gain: +0.3-0.6% accuracy

[OK] 22. Cross-Mission Feature Standardization
   - Normalize features across missions
   - Handle mission-specific biases
   - Expected gain: +0.2-0.5% accuracy
""")
    
    print(f"\n{'=' * 80}")
    print("RECOMMENDED IMMEDIATE ACTIONS (Quick Wins)")
    print("=" * 80)
    
    print(f"""
[START] TOP 5 QUICK WINS (Can implement today, high ROI):

1.  ADD XGBOOST TO ENSEMBLE [Expected: +1-2%]
   pip install xgboost
   Add XGBClassifier(scale_pos_weight=imbalance_ratio)

2.  IMPLEMENT SMOTE FOR CLASS BALANCE [Expected: +0.5-1.5%]
   pip install imbalanced-learn
   from imblearn.over_sampling import SMOTE
   X_train, y_train = SMOTE().fit_resample(X_train, y_train)

3.  CREATE DOMAIN-SPECIFIC FEATURES [Expected: +0.5-1%]
   - transit_depth = (planet_radius / star_radius)²
   - orbital_velocity = 2π × semi_major_axis / period
   - snr_features from error columns

4.  OPTIMIZE PROBABILITY THRESHOLD [Expected: +0.3-0.8%]
   from sklearn.metrics import f1_score
   Test thresholds from 0.3 to 0.7, pick best F1

5.  USE STACKING ENSEMBLE [Expected: +0.5-1.5%]
   from sklearn.ensemble import StackingClassifier
   Stack RF + GB + XGB, use LR as meta-learner

[INFO] TOTAL EXPECTED GAIN FROM TOP 5: +3-7% accuracy
   → Could reach 72.5-76.5% accuracy!
""")
    
    print(f"\n{'=' * 80}")
    print("IMPLEMENTATION PRIORITY")
    print("=" * 80)
    
    print(f"""
Phase 1 (Today - 2 hours):
  [OK] Install XGBoost and LightGBM
  [OK] Add to ensemble with optimized parameters
  [OK] Implement SMOTE oversampling
  [OK] Test on validation set
  Expected: 71-73% accuracy

Phase 2 (Next day - 3 hours):
  [OK] Create domain-specific astronomical features
  [OK] Implement stacking ensemble
  [OK] Optimize probability threshold
  [OK] Full training and evaluation
  Expected: 73-75% accuracy

Phase 3 (If time allows - 4+ hours):
  [OK] Bayesian hyperparameter optimization
  [OK] Neural network classifier
  [OK] Advanced feature interactions
  [OK] Comprehensive ensemble
  Expected: 75-78% accuracy

[TARGET] REALISTIC TARGET: 73-76% accuracy within 1-2 days
[STAR] STRETCH GOAL: 77-80% accuracy with deep learning (1 week)
""")

if __name__ == "__main__":
    model_data = analyze_current_model()
    identify_improvement_opportunities(model_data)
    
    print(f"\n{'=' * 80}")
    print("Analysis complete! Ready to implement improvements.")
    print("=" * 80)
