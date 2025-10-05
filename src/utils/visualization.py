"""
Visualization utilities for exoplanet detection pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Any
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    title: str = "Light Curve",
    xlabel: str = "Time (days)",
    ylabel: str = "Normalized Flux",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    show_transit: bool = False,
    transit_times: Optional[List[float]] = None
) -> plt.Figure:
    """Plot a light curve.
    
    Args:
        time: Time array
        flux: Flux array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        show_transit: Whether to highlight transit times
        transit_times: List of transit times to highlight
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot light curve
    ax.plot(time, flux, 'b-', alpha=0.7, linewidth=0.8)
    
    # Highlight transits if provided
    if show_transit and transit_times:
        for transit_time in transit_times:
            ax.axvline(transit_time, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_phase_folded_curve(
    phase: np.ndarray,
    flux: np.ndarray,
    period: float,
    title: str = "Phase-Folded Light Curve",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot a phase-folded light curve.
    
    Args:
        phase: Phase array
        flux: Flux array
        period: Orbital period
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by phase for better visualization
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]
    
    ax.plot(phase_sorted, flux_sorted, 'b-', alpha=0.7, linewidth=1.0)
    ax.scatter(phase_sorted, flux_sorted, c='blue', s=2, alpha=0.5)
    
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Normalized Flux', fontsize=12)
    ax.set_title(f"{title} (Period: {period:.4f} days)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at phase 0 (transit center)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Transit Center')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
    normalize: bool = False
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        normalize: Whether to normalize the matrix
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if labels is None:
        labels = ['False Positive', 'Exoplanet']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: dict,
    metrics: List[str] = ['loss', 'accuracy', 'f1'],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot training history.
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}')
        
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} History')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    features: List[str],
    importance: np.ndarray,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    top_n: Optional[int] = None
) -> plt.Figure:
    """Plot feature importance.
    
    Args:
        features: Feature names
        importance: Feature importance values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        top_n: Number of top features to show
        
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_idx = np.argsort(importance)[::-1]
    
    if top_n:
        sorted_idx = sorted_idx[:top_n]
    
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(range(len(sorted_features)), sorted_importance)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    title: str = "Interactive Light Curve",
    save_path: Optional[Path] = None
) -> go.Figure:
    """Create interactive light curve plot with Plotly.
    
    Args:
        time: Time array
        flux: Flux array
        title: Plot title
        save_path: Path to save HTML file
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=flux,
        mode='lines',
        name='Light Curve',
        line=dict(width=1, color='blue')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (days)',
        yaxis_title='Normalized Flux',
        hovermode='x unified',
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig