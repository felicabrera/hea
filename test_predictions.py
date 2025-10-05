"""
Test predictions directly with the sample CSV
"""
import sys
import io
# Set UTF-8 encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import joblib
import numpy as np
import argparse
from pathlib import Path
import shap
import warnings
warnings.filterwarnings('ignore')

def convert_ra_to_decimal(ra_str):
    """Convert RA string like '19:02:43.1' to decimal degrees"""
    try:
        if pd.isna(ra_str) or ra_str == '':
            return 0.0
        if isinstance(ra_str, (int, float)):
            return float(ra_str)
        parts = str(ra_str).split(':')
        hours = float(parts[0])
        minutes = float(parts[1]) if len(parts) > 1 else 0
        seconds = float(parts[2]) if len(parts) > 2 else 0
        decimal_degrees = (hours + minutes/60 + seconds/3600) * 15  # Convert hours to degrees
        return decimal_degrees
    except:
        return 0.0

def preprocess_data(df):
    """Convert any string columns to numeric"""
    print("\nPreprocessing data...")
    
    # Check for string columns
    string_cols = df.select_dtypes(include=['object']).columns
    print(f"Found {len(string_cols)} string columns: {list(string_cols)[:10]}")
    
    for col in string_cols:
        # Special handling for RA string format
        if 'rastr' in col.lower() or col == 'rastr':
            print(f"Converting {col} from HMS format to decimal degrees...")
            df[col] = df[col].apply(convert_ra_to_decimal)
        else:
            # Try to convert to numeric, fill with 0 if fails
            print(f"Converting {col} to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Warning: {col} still contains strings, forcing to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def load_model():
    """Load the trained model"""
    print("Loading model...")
    model_dir = Path("models/catalog_models")
    models = list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pkl"))
    
    if not models:
        raise FileNotFoundError("No model files found!")
    
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_model.name}")
    
    model_data = joblib.load(latest_model)
    
    # Extract the actual model object
    if 'stacking' in model_data and hasattr(model_data['stacking'], 'predict'):
        model = model_data['stacking']
        print("Using 'stacking' model")
    elif 'model' in model_data and isinstance(model_data['model'], dict):
        if 'stacking' in model_data['model']:
            model = model_data['model']['stacking']
            print("Using nested 'model->stacking' model")
        else:
            model = model_data['model']
            print("Using 'model' object")
    elif 'mlp' in model_data:
        model = model_data['mlp']
        print("Using 'mlp' model")
    else:
        raise ValueError("Could not find model object")
    
    feature_names = model_data.get('feature_names', [])
    threshold = model_data.get('threshold', 0.5)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"Threshold: {threshold}")
    
    return model, feature_names, threshold

def get_base_estimators(model):
    """Extract base estimators from StackingClassifier"""
    try:
        if hasattr(model, 'estimators_'):
            estimator_names = []
            estimators = []
            if hasattr(model, 'named_estimators_'):
                for name, est in model.named_estimators_.items():
                    estimator_names.append(name)
                    estimators.append(est)
            else:
                for i, est in enumerate(model.estimators_):
                    estimator_names.append(f"Model_{i+1}")
                    estimators.append(est)
            return estimator_names, estimators
        else:
            return [], []
    except Exception as e:
        print(f"Warning: Could not extract base estimators: {e}")
        return [], []

def calculate_explanations(model, X, feature_names, n_samples=100):
    """Calculate SHAP explanations for predictions"""
    try:
        print("\nCalculating explanations (this may take a moment)...")
        X_explain = X.iloc[:n_samples] if len(X) > n_samples else X
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        except Exception as e:
            print(f"   Using KernelExplainer (slower but works with any model)...")
            background = shap.sample(X_explain, min(50, len(X_explain)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        return shap_values, explainer
    except Exception as e:
        print(f"WARNING: Could not calculate SHAP values: {e}")
        return None, None

def format_explanation(row_idx, X, shap_values, feature_names, prediction, probability, show_top_n=5):
    """Format explanation text for a single prediction"""
    output = []
    output.append(f"\n{'─'*70}")
    output.append(f"EXPLANATION FOR ROW {row_idx + 1}")
    output.append(f"{'─'*70}")
    if shap_values is not None:
        row_shap = shap_values[row_idx] if row_idx < len(shap_values) else None
        if row_shap is not None:
            # Flatten if needed
            row_shap = np.array(row_shap).flatten()
            shap_importance = np.abs(row_shap)
            top_indices = np.argsort(shap_importance)[-show_top_n:][::-1]
            output.append(f"\n[KEY] KEY EVIDENCE (Top {show_top_n} Features):\n")
            for i in range(len(top_indices)):
                idx = int(top_indices[i])
                feature_name = feature_names[idx]
                feature_value = float(X.iloc[row_idx, idx])
                shap_value = float(row_shap[idx])
                if shap_value > 0:
                    direction = "[OK] Supports EXOPLANET"
                    symbol = "+"
                else:
                    direction = "[FAIL] Against EXOPLANET"
                    symbol = ""
                output.append(f"  {i+1}. {feature_name}")
                output.append(f"     Value: {feature_value:.6f}")
                output.append(f"     Impact: {symbol}{shap_value:.4f} ({direction})")
                output.append("")
            output.append(f"\n[INFO] FINAL DECISION:")
            if prediction == 1:
                output.append(f"   → EXOPLANET detected with {probability*100:.1f}% confidence")
            else:
                output.append(f"   → NOT an exoplanet ({(1-probability)*100:.1f}% confidence)")
    return "\n".join(output)

def make_predictions(csv_path, custom_threshold=None, explain=False):
    """Make predictions on CSV file"""
    print(f"\n{'='*60}")
    print(f"EXOPLANET PREDICTION TEST")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Load model
    model, feature_names, threshold = load_model()
    
    # Preprocess data (convert strings to numeric)
    df = preprocess_data(df)
    
    # Check for missing features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        print(f"\nWarning: Adding {len(missing_features)} missing features with zeros")
        for feat in missing_features:
            df[feat] = 0.0
    
    # Select only required features
    X = df[feature_names].copy()
    
    # Verify all numeric
    print(f"\nData types check:")
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"ERROR: Non-numeric columns found: {list(non_numeric)}")
        for col in non_numeric:
            print(f"  {col}: {X[col].dtype} - Sample: {X[col].iloc[0]}")
        raise ValueError("Data contains non-numeric values")
    else:
        print(" All columns are numeric")
    
    # Check for NaN values and fill them
    nan_counts = X.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"\nWarning: Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
        print(f"Filling NaN values with 0...")
        X = X.fillna(0)
    
    # Check for infinity values
    inf_mask = np.isinf(X.select_dtypes(include=[np.number]))
    if inf_mask.any().any():
        print(f"Warning: Found infinity values, replacing with 0...")
        X = X.replace([np.inf, -np.inf], 0)
    
    print(f"\nMaking predictions...")
    print(f"Input shape: {X.shape}")
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = model.predict(X)
    
    # Use custom threshold if provided
    original_threshold = threshold
    if custom_threshold is not None:
        threshold = custom_threshold
        print(f"\nUsing custom threshold: {threshold:.2f} (model optimal: {original_threshold:.2f})")
        if threshold < original_threshold:
            print("   → More sensitive: Will detect more exoplanets but may include more false positives")
        elif threshold > original_threshold:
            print("   → More conservative: Fewer false positives but may miss some exoplanets")
        else:
            print("   → Using model's optimal threshold")
    
    predictions = (probabilities >= threshold).astype(int)
    
    # Calculate explanations if requested
    shap_values = None
    explainer = None
    estimator_names = []
    estimator_predictions = []
    
    if explain:
        estimator_names, estimators = get_base_estimators(model)
        if estimators:
            print(f"\n Calculating individual model predictions...")
            for name, estimator in zip(estimator_names, estimators):
                try:
                    if hasattr(estimator, 'predict_proba'):
                        est_proba = estimator.predict_proba(X)[:, 1]
                        estimator_predictions.append((name, est_proba))
                except Exception as e:
                    print(f"   Warning: Could not get predictions from {name}: {e}")
        shap_values, explainer = calculate_explanations(model, X, feature_names)
    
    # Create results
    results = pd.DataFrame({
        'Row': range(1, len(df) + 1),
        'Prediction': ['EXOPLANET' if p == 1 else 'NOT EXOPLANET' for p in predictions],
        'Confidence': [f"{p*100:.1f}%" if pred == 1 else f"{(1-p)*100:.1f}%" 
                      for p, pred in zip(probabilities, predictions)],
        'Probability': probabilities,
        'Raw_Prediction': predictions
    })
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    print(results.to_string(index=False))
    
    # Print explanations if requested
    if explain and shap_values is not None:
        print(f"\n\n{'='*70}")
        print("EXPLAINABLE AI - PREDICTION EXPLANATIONS")
        print(f"{'='*70}")
        if estimator_predictions:
            print(f"\nENSEMBLE MODEL AGREEMENT:\n")
            for row_idx in range(len(X)):
                print(f"Row {row_idx + 1}:")
                for est_name, est_proba in estimator_predictions:
                    conf = est_proba[row_idx] * 100
                    print(f"  {est_name:<20}: {conf:>5.1f}% confidence")
                print()
        for row_idx in range(len(X)):
            explanation_text = format_explanation(
                row_idx, X, shap_values, feature_names,
                predictions[row_idx], probabilities[row_idx]
            )
            print(explanation_text)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Threshold used: {threshold:.2f}")
    print(f"Total predictions: {len(results)}")
    print(f"Exoplanets detected: {(predictions == 1).sum()}")
    print(f"Not exoplanets: {(predictions == 0).sum()}")
    print(f"Average confidence: {probabilities.mean()*100:.1f}%")
    print(f"Min confidence: {probabilities.min()*100:.1f}%")
    print(f"Max confidence: {probabilities.max()*100:.1f}%")
    
    # Show comparison with optimal threshold if custom threshold was used
    if custom_threshold is not None and custom_threshold != original_threshold:
        optimal_predictions = (probabilities >= original_threshold).astype(int)
        optimal_exoplanets = (optimal_predictions == 1).sum()
        print(f"\nComparison with optimal threshold ({original_threshold:.2f}):")
        print(f"   Optimal would detect: {optimal_exoplanets} exoplanets")
        print(f"   Difference: {(predictions == 1).sum() - optimal_exoplanets:+d} exoplanets")
    
    # Save results
    output_file = csv_path.replace('.csv', '_predictions.csv')
    
    # Add all original columns plus predictions
    output_df = df.copy()
    output_df['Prediction'] = results['Prediction']
    output_df['Confidence'] = results['Confidence']
    output_df['Probability'] = results['Probability']
    
    output_df.to_csv(output_file, index=False)
    print(f"\n Results saved to: {output_file}")
    
    return results

def compare_thresholds(csv_path, thresholds=[0.2, 0.3, 0.38, 0.4, 0.5, 0.6, 0.7]):
    """Compare predictions across multiple thresholds"""
    print(f"\n{'='*60}")
    print(f"THRESHOLD COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    # Load data and model once
    df = pd.read_csv(csv_path)
    df = preprocess_data(df)
    model, feature_names, optimal_threshold = load_model()
    
    # Check for missing features
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        for feat in missing_features:
            df[feat] = 0.0
    
    X = df[feature_names].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Get probabilities
    probabilities = model.predict_proba(X)[:, 1]
    
    # Compare thresholds
    print(f"\nAnalyzing {len(thresholds)} different thresholds...\n")
    print(f"{'Threshold':<12} {'Exoplanets':<12} {'Not Exo':<12} {'Detection Rate':<15}")
    print("-" * 60)
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        n_exo = (predictions == 1).sum()
        n_not = (predictions == 0).sum()
        rate = n_exo / len(predictions) * 100
        
        marker = " ← OPTIMAL" if abs(threshold - optimal_threshold) < 0.01 else ""
        print(f"{threshold:<12.2f} {n_exo:<12} {n_not:<12} {rate:<14.1f}%{marker}")
    
    print(f"\nTips for choosing threshold:")
    print(f"   • Lower (0.2-0.3): Best for discovery missions, maximize detections")
    print(f"   • Optimal ({optimal_threshold:.2f}): Balanced precision and recall")
    print(f"   • Higher (0.5-0.7): Best for follow-up, minimize false positives")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Make exoplanet predictions with optional threshold tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default threshold (model optimal)
  python test_predictions.py sample_predictions_input.csv
  
  # Use custom threshold (more sensitive)
  python test_predictions.py sample_predictions_input.csv --threshold 0.3
  
  # Use custom threshold (more conservative)
  python test_predictions.py sample_predictions_input.csv -t 0.5
  
  # Compare multiple thresholds
  python test_predictions.py sample_predictions_input.csv --compare
        """
    )
    
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='sample_predictions_input.csv',
        help='Path to CSV file with candidate data (default: sample_predictions_input.csv)'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=None,
        help='Custom classification threshold (0.0-1.0). Lower = more sensitive, Higher = more conservative'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare predictions across multiple thresholds'
    )
    
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Show detailed explanations using SHAP values (slower but provides insights)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Run threshold comparison
            compare_thresholds(args.csv_path)
        else:
            # Run single prediction
            results = make_predictions(args.csv_path, args.threshold, args.explain)
            print("\n Prediction test completed successfully!")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
