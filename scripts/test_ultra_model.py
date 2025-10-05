"""
Test the Ultra Model and Compare with Baseline

This script loads the trained ultra model and evaluates its performance
on the test set, comparing it with the baseline model.

Date: October 4, 2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """Load the trained model."""
    print(f"\n Loading model from: {model_path}")
    model_data = joblib.load(model_path)
    print("[OK] Model loaded successfully!")
    
    # Display model info
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    
    if 'metadata' in model_data:
        metadata = model_data['metadata']
        print(f"Training Date: {metadata.get('training_date', 'N/A')}")
        print(f"Phases Trained: {metadata.get('phases', 'N/A')}")
        print(f"Training Samples: {metadata.get('n_samples', 'N/A')}")
        print(f"Features Used: {metadata.get('n_features', 'N/A')}")
        print(f"Optimal Threshold: {metadata.get('optimal_threshold', 'N/A')}")
        
        if 'training_metrics' in metadata:
            print("\n[DATA] Training Performance:")
            metrics = metadata['training_metrics']
            print(f"  Accuracy:  {metrics.get('accuracy', 0)*100:.2f}%")
            print(f"  Precision: {metrics.get('precision', 0)*100:.2f}%")
            print(f"  Recall:    {metrics.get('recall', 0)*100:.2f}%")
            print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
            print(f"  AUC:       {metrics.get('auc', 0):.4f}")
    
    return model_data


def predict_with_model(model_data, X):
    """Make predictions using the ultra model."""
    model = model_data['final_ensemble']
    threshold = model_data['metadata'].get('optimal_threshold', 0.5)
    
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        y_proba = None
    
    return y_pred, y_proba


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """Evaluate model performance."""
    print("\n" + "="*80)
    print(f"[TARGET] {model_name.upper()} PERFORMANCE")
    print("="*80)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"AUC Score: {auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n[DATA] Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"Actual 0      {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual 1      {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Calculate rates
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    print(f"\n[UP] Classification Rates:")
    print(f"True Positive Rate (TPR):  {tpr*100:.2f}% (Caught exoplanets)")
    print(f"True Negative Rate (TNR):  {tnr*100:.2f}% (Rejected false positives)")
    print(f"False Positive Rate (FPR): {fpr*100:.2f}% (False alarms)")
    print(f"False Negative Rate (FNR): {fnr*100:.2f}% (Missed exoplanets)")
    
    # Detailed classification report
    print("\n Detailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                               target_names=['False Positive', 'Exoplanet'],
                               digits=4))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc if y_proba is not None else None,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False Positive', 'Exoplanet'],
                yticklabels=['False Positive', 'Exoplanet'],
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[DATA] Confusion matrix saved to: {save_path}")
    
    plt.close()


def compare_models(baseline_metrics, ultra_metrics):
    """Compare baseline and ultra model performance."""
    print("\n" + "="*80)
    print("[DATA] BASELINE vs ULTRA MODEL COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Ultra Model':<15} {'Improvement':<15}")
    print("-" * 65)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric)
        ultra_val = ultra_metrics.get(metric)
        
        if baseline_val is not None and ultra_val is not None:
            if metric == 'auc':
                baseline_str = f"{baseline_val:.4f}"
                ultra_str = f"{ultra_val:.4f}"
                improvement = ultra_val - baseline_val
                improvement_str = f"{improvement:+.4f}"
            else:
                baseline_str = f"{baseline_val*100:.2f}%"
                ultra_str = f"{ultra_val*100:.2f}%"
                improvement = (ultra_val - baseline_val) * 100
                improvement_str = f"{improvement:+.2f}%"
            
            print(f"{metric.capitalize():<20} {baseline_str:<15} {ultra_str:<15} {improvement_str:<15}")
    
    # Overall summary
    acc_improvement = (ultra_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    
    print("\n" + "="*80)
    if acc_improvement > 0:
        print(f"[CELEBRATE] SUCCESS! Ultra model improved accuracy by {acc_improvement:+.2f} percentage points!")
        if ultra_metrics['accuracy'] >= 0.75:
            print(f"[OK] TARGET ACHIEVED! Accuracy: {ultra_metrics['accuracy']*100:.2f}% (≥75% goal)")
        else:
            print(f"[UP] Progress made! Accuracy: {ultra_metrics['accuracy']*100:.2f}% (goal: ≥75%)")
    else:
        print(f"WARNING:  Ultra model accuracy change: {acc_improvement:+.2f} percentage points")
    print("="*80)


def main():
    """Main testing function."""
    print("\n" + "="*80)
    print("[TEST] ULTRA MODEL TESTING")
    print("="*80)
    
    # Find the latest ultra model
    models_dir = Path('models/catalog_models')
    ultra_models = list(models_dir.glob('ultra_model_all_phases_*.joblib'))
    
    if not ultra_models:
        print("[FAIL] No ultra model found!")
        print("Expected location: models/catalog_models/ultra_model_all_phases_*.joblib")
        return
    
    # Get the most recent model
    latest_model = max(ultra_models, key=lambda p: p.stat().st_mtime)
    
    # Load the ultra model
    ultra_model_data = load_model(latest_model)
    
    # Check if we have test data in the model
    if 'X_test' in ultra_model_data and 'y_test' in ultra_model_data:
        print("\n[OK] Test data found in model file!")
        X_test = ultra_model_data['X_test']
        y_test = ultra_model_data['y_test']
        print(f"Test samples: {len(X_test)}")
        print(f"Test labels: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")
    else:
        print("\nWARNING:  No test data found in model file.")
        print("The model was evaluated during training. Check the training logs for results.")
        return
    
    # Make predictions
    print("\n Making predictions on test set...")
    y_pred, y_proba = predict_with_model(ultra_model_data, X_test)
    print(f"Predictions generated: {len(y_pred)}")
    
    # Evaluate ultra model
    ultra_metrics = evaluate_model(y_test, y_pred, y_proba, "Ultra Model")
    
    # Try to load baseline model for comparison
    baseline_models = list(models_dir.glob('best_voting_model_*.joblib'))
    
    if baseline_models:
        print("\n" + "="*80)
        print("[DATA] LOADING BASELINE MODEL FOR COMPARISON")
        print("="*80)
        
        latest_baseline = max(baseline_models, key=lambda p: p.stat().st_mtime)
        print(f"Baseline model: {latest_baseline.name}")
        
        try:
            baseline_data = joblib.load(latest_baseline)
            baseline_model = baseline_data.get('model')
            
            if baseline_model:
                print("[OK] Baseline model loaded!")
                
                # Make baseline predictions
                if hasattr(baseline_model, 'predict_proba'):
                    y_pred_baseline = baseline_model.predict(X_test)
                    y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_baseline = baseline_model.predict(X_test)
                    y_proba_baseline = None
                
                # Evaluate baseline
                baseline_metrics = evaluate_model(y_test, y_pred_baseline, y_proba_baseline, "Baseline Model")
                
                # Compare models
                compare_models(baseline_metrics, ultra_metrics)
                
                # Plot confusion matrices
                fig_dir = Path('docs')
                fig_dir.mkdir(exist_ok=True)
                
                plot_confusion_matrix(baseline_metrics['confusion_matrix'], 
                                    'Baseline Model',
                                    fig_dir / 'baseline_confusion_matrix.png')
                plot_confusion_matrix(ultra_metrics['confusion_matrix'], 
                                    'Ultra Model',
                                    fig_dir / 'ultra_confusion_matrix.png')
        except Exception as e:
            print(f"WARNING:  Could not load baseline model: {e}")
    else:
        print("\nWARNING:  No baseline model found for comparison.")
        print("Ultra model performance is shown above.")
    
    # Save test results
    results_file = Path('experiments/runs') / f'ultra_model_test_results_{latest_model.stem.split("_")[-1]}.txt'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("ULTRA MODEL TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {latest_model.name}\n")
        f.write(f"Test Date: {pd.Timestamp.now()}\n\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"Test Distribution: 0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}\n\n")
        f.write(f"Accuracy:  {ultra_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {ultra_metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:    {ultra_metrics['recall']*100:.2f}%\n")
        f.write(f"F1 Score:  {ultra_metrics['f1']:.4f}\n")
        if ultra_metrics['auc']:
            f.write(f"AUC Score: {ultra_metrics['auc']:.4f}\n")
    
    print(f"\n Test results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("[OK] TESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
