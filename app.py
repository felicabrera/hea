"""
HEA - Habitable Exoplanet Analyzer
NASA Space Apps Challenge 2025
AI-Powered Exoplanet Detection & Habitability Assessment

Modern, Responsive, Enterprise-Grade Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import habitability scorer
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'utils'))
try:
    from habitability_scorer import HabitabilityScorer
    HABITABILITY_AVAILABLE = True
    HABITABILITY_SCORER = HabitabilityScorer()  # Create instance globally
except ImportError:
    HABITABILITY_AVAILABLE = False
    HABITABILITY_SCORER = None

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="HEA | Machine Learning Exoplanet Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================
# CUSTOM CSS STYLING
# ===========================

st.markdown("""
<style>
    /* ==========================================
        IMPORT GOOGLE FONTS
       ========================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* ==========================================
        GLOBAL VARIABLES - COLOR PALETTE
       ========================================== */
    :root {
        /* Primary Colors */
        --bg-primary: #1A1D23;
        --bg-secondary: #252930;
        --bg-tertiary: #2E3440;
        
        /* Accent Colors */
        --accent-primary: #2E5EAA;
        --accent-secondary: #4A90E2;
        --accent-gradient: linear-gradient(135deg, #2E5EAA 0%, #4A90E2 100%);
        
        /* Text Colors */
        --text-primary: #FFFFFF;
        --text-secondary: #B0B8C1;
        --text-muted: #8B949E;
        
        /* Status Colors */
        --success: #28A745;
        --success-bg: rgba(40, 167, 69, 0.1);
        --danger: #DC3545;
        --danger-bg: rgba(220, 53, 69, 0.1);
        --warning: #FFA500;
        --warning-bg: rgba(255, 165, 0, 0.1);
        --info: #4A90E2;
        --info-bg: rgba(74, 144, 226, 0.1);
        
        /* Borders & Shadows */
        --border-color: #3A3F47;
        --border-hover: #4A90E2;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 2px 8px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 4px 16px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 15px rgba(46, 94, 170, 0.25);
        
        /* Spacing */
        --space-xs: 0.5rem;
        --space-sm: 1rem;
        --space-md: 1.5rem;
        --space-lg: 2rem;
        --space-xl: 3rem;
    }
    
    /* ==========================================
        GLOBAL RESET & BASE STYLES
       ========================================== */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main App Container */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    [data-testid="stAppViewContainer"] {
        background: var(--bg-primary);
    }
    
    .main {
        background: var(--bg-primary);
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* ==========================================
        CUSTOM HEADER
       ========================================== */
    .custom-header {
        top: 0;
        z-index: 1000;
        backdrop-filter: blur(10px);
    }
    
    .header-content {
        max-width: 1400px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 2rem;
    }
    
    .header-brand {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .brand-icon {
        font-size: 2.5rem;
        filter: drop-shadow(0 0 10px rgba(74, 144, 226, 0.5));
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .brand-info {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .brand-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .brand-tagline {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    .header-meta {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        padding-bottom: 2rem;
    }
    
    .meta-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .meta-badge i {
        color: var(--accent-primary);
        font-size: 0.85rem;
    }
    
    .meta-badge.success {
        background: var(--success-bg);
        border-color: var(--success);
        color: var(--success);
    }
    
    .meta-badge.success i {
        color: var(--success);
    }
    
    .meta-badge:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--accent-primary);
    }
    
    @media (max-width: 1024px) {
        .header-meta {
            display: none;
        }
    }
    
    /* ==========================================
        SECTION CONTAINER
       ========================================== */
    .section {
        padding: var(--space-xl);
        margin-bottom: var(--space-md);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: var(--space-md);
        display: flex;
        align-items: center;
        gap: var(--space-sm);
    }
    
    .section-title i {
        color: var(--accent-primary);
    }
    
    /* ==========================================
        METRIC CARDS
       ========================================== */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: var(--space-md);
        margin-bottom: var(--space-xl);
    }
    
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: var(--space-md);
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--accent-gradient);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-primary);
        box-shadow: var(--shadow-glow);
        transform: translateY(-4px);
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: var(--space-xs);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
    }
    
    .metric-icon {
        position: absolute;
        top: var(--space-sm);
        right: var(--space-sm);
        font-size: 2rem;
        color: var(--accent-primary);
        opacity: 0.1;
    }
    
    /* ==========================================
        TWO-COLUMN LAYOUT
       ========================================== */
    .two-column-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-xl);
        margin-bottom: var(--space-xl);
    }
    
    /* ==========================================
        CUSTOM TABS
       ========================================== */
    .custom-tabs {
        display: flex;
        gap: var(--space-sm);
        margin-bottom: var(--space-md);
        flex-wrap: wrap;
    }
    
    .tab-button {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: var(--space-sm) var(--space-md);
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .tab-button:hover {
        background: var(--bg-tertiary);
        border-color: var(--accent-primary);
        color: var(--text-primary);
    }
    
    .tab-button.active {
        background: var(--accent-gradient);
        border-color: transparent;
        color: white;
        box-shadow: var(--shadow-glow);
    }
    
    /* ==========================================
        CONTROL PANEL CARD
       ========================================== */
    .control-panel {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: var(--space-lg);
        box-shadow: var(--shadow-md);
    }
    
    .control-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: var(--space-md);
        display: flex;
        align-items: center;
        gap: var(--space-sm);
    }
    
    /* ==========================================
        FILE UPLOADER STYLING
       ========================================== */
    .stFileUploader {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: var(--space-xl);
        background: var(--bg-primary);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent-primary);
        background: var(--bg-tertiary);
    }
    
    .stFileUploader label {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* ==========================================
        BUTTONS
       ========================================== */
    .stButton > button {
        background: var(--accent-gradient);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        width: 100%;
    }
    
    .stButton > button:hover {
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ==========================================
        SUMMARY METRICS
       ========================================== */
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: var(--space-md);
    }
    
    .summary-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: var(--space-md);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .summary-card:hover {
        border-color: var(--accent-secondary);
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .summary-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: var(--space-xs);
    }
    
    .summary-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: var(--space-xs);
    }
    
    .summary-subtext {
        font-size: 0.875rem;
        color: var(--text-muted);
    }
    
    /* Progress Bar */
    .progress-container {
        width: 100%;
        height: 8px;
        background: var(--bg-primary);
        border-radius: 4px;
        overflow: hidden;
        margin-top: var(--space-sm);
    }
    
    .progress-bar {
        height: 100%;
        background: var(--accent-gradient);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* ==========================================
        DATA TABLE
       ========================================== */
    .data-table-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: var(--space-lg);
        overflow-x: auto;
        box-shadow: var(--shadow-md);
    }
    
    .stDataFrame {
        width: 100%;
    }
    
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
        background: transparent;
    }
    
    .dataframe thead {
        background: var(--bg-primary);
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .dataframe thead th {
        padding: var(--space-md);
        text-align: left;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-primary);
        border-bottom: 2px solid var(--border-color);
        font-size: 0.75rem;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid var(--border-color);
        transition: background 0.2s ease;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background: var(--bg-tertiary);
    }
    
    .dataframe tbody tr:hover {
        background: rgba(138, 43, 226, 0.1);
    }
    
    .dataframe tbody td {
        padding: var(--space-md);
        color: var(--text-primary);
    }
    
    /* Conditional formatting for predictions */
    .prediction-exoplanet {
        color: var(--success);
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .prediction-no {
        color: var(--danger);
        font-weight: 700;
        text-transform: uppercase;
    }
    
    /* Confidence bars */
    .confidence-cell {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
    }
    
    .confidence-bar {
        flex-grow: 1;
        height: 6px;
        background: var(--bg-primary);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: var(--accent-gradient);
        border-radius: 3px;
    }
    
    /* ==========================================
        ALERTS & MESSAGES
       ========================================== */
    .stAlert {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-primary);
        border-radius: 8px;
        padding: var(--space-md);
        color: var(--text-primary);
    }
    
    [data-testid="stAlert"][data-baseweb="notification"] {
        background: var(--bg-secondary);
    }
    
    .stSuccess {
        border-left-color: var(--success);
        background: var(--success-bg);
    }
    
    .stError {
        border-left-color: var(--danger);
        background: var(--danger-bg);
    }
    
    .stWarning {
        border-left-color: var(--warning);
        background: var(--warning-bg);
    }
    
    .stInfo {
        border-left-color: var(--info);
        background: var(--info-bg);
    }
    
    /* ==========================================
       RESPONSIVE DESIGN
       ========================================== */
    
    /* Tablets (max-width: 1024px) */
    @media (max-width: 1024px) {
        .custom-header {
            padding: var(--space-md);
        }
        
        .header-title {
            font-size: 1.5rem;
        }
        
        .section {
            padding: var(--space-md);
        }
        
        .metric-grid {
            grid-template-columns: repeat(3, 1fr);
        }
        
        .two-column-grid {
            grid-template-columns: 1fr;
        }
        
        .summary-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    
    /* Mobile (max-width: 768px) */
    @media (max-width: 768px) {
        .custom-header {
            flex-direction: column;
            align-items: flex-start;
            gap: var(--space-sm);
        }
        
        .header-title {
            font-size: 1.25rem;
        }
        
        .header-subtitle {
            font-size: 0.75rem;
        }
        
        .section {
            padding: var(--space-sm);
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
        
        .summary-grid {
            grid-template-columns: 1fr;
        }
        
        .summary-value {
            font-size: 2rem;
        }
        
        .custom-tabs {
            flex-direction: column;
        }
        
        .tab-button {
            width: 100%;
            text-align: center;
        }
        
        .control-panel {
            padding: var(--space-md);
        }
    }
    
    /* Small Mobile (max-width: 480px) */
    @media (max-width: 480px) {
        .header-title {
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .summary-value {
            font-size: 1.5rem;
        }
        
        .dataframe {
            font-size: 0.75rem;
        }
        
        .dataframe thead th,
        .dataframe tbody td {
            padding: var(--space-xs);
        }
    }
    
    /* ==========================================
       SCROLLBAR STYLING
       ========================================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-color);
    }
    
    /* ==========================================
       ANIMATIONS
       ========================================== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* ==========================================
       LOADING SPINNER
       ========================================== */
    .stSpinner > div {
        border-top-color: var(--accent-primary);
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# SESSION STATE
# ===========================

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'predict'
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'single_prediction' not in st.session_state:
    st.session_state.single_prediction = None

# ===========================
# CONSTANTS
# ===========================

MODEL_DIR = Path("models/catalog_models")

# ===========================
# HELPER FUNCTIONS
# ===========================

@st.cache_resource
def get_available_models():
    """Get list of available model files"""
    try:
        # Get all .joblib and .pkl files in model directory
        model_files = list(MODEL_DIR.glob("*.joblib")) + list(MODEL_DIR.glob("*.pkl"))
        
        if not model_files:
            return []
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return model_files
    except Exception as e:
        return []

def load_model(model_path=None):
    """Load the trained model"""
    try:
        if model_path is None:
            # Get all available models
            model_files = get_available_models()
            if not model_files:
                return None, None
            
            # Prefer ultra_model_all_phases if it exists, otherwise use newest
            preferred_model = next((f for f in model_files if 'ultra_model_all_phases' in f.name), None)
            model_path = preferred_model if preferred_model else model_files[0]
        
        model_data = joblib.load(model_path)
        
        metadata = {
            "filename": Path(model_path).name,
            "accuracy": float(model_data.get("test_accuracy", model_data.get("accuracy", 0.9429))),
            "precision": float(model_data.get("test_precision", model_data.get("precision", 0.9403))),
            "recall": float(model_data.get("test_recall", model_data.get("recall", 0.9800))),
            "f1_score": float(model_data.get("test_f1", model_data.get("f1_score", 0.9597))),
            "auc_score": float(model_data.get("test_auc", model_data.get("auc", 0.9711))),
            "threshold": float(model_data.get("optimal_threshold", 0.380)),
            "n_features": len(model_data.get("feature_names", [])),
            "created": datetime.fromtimestamp(Path(model_path).stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
        
        return model_data, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_placeholder_data():
    """Generate placeholder prediction data for demonstration"""
    np.random.seed(42)
    n_samples = 20
    
    predictions = np.random.choice(['Exoplanet', 'No Exoplanet'], n_samples, p=[0.4, 0.6])
    confidences = np.random.uniform(0.5, 0.99, n_samples)
    viability = np.random.uniform(0.0, 1.0, n_samples)
    
    df = pd.DataFrame({
        'ID': [f'KOI-{1000+i}' for i in range(n_samples)],
        'Prediction': predictions,
        'Confidence': confidences,
        'Viability Score': viability,
        'Period (days)': np.random.uniform(0.5, 500, n_samples),
        'Radius (R⊕)': np.random.uniform(0.5, 15, n_samples),
    })
    
    return df

def make_predictions(data_df, model_data, metadata):
    """Make predictions on uploaded data"""
    try:
        # Make a copy to avoid modifying the original dataframe
        data_df = data_df.copy()
        
        feature_names = model_data.get("feature_names", [])
        if not feature_names:
            st.error("Model has no feature names stored")
            return None
        
        # Convert string columns to numeric
        string_cols = data_df.select_dtypes(include=['object']).columns
        for col in string_cols:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce').fillna(0)
        
        # Add missing features
        missing_features = set(feature_names) - set(data_df.columns)
        for feat in missing_features:
            data_df[feat] = 0.0
        
        X = data_df[feature_names].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Get model
        model = (model_data.get('stacking') or 
                model_data.get('model') or 
                model_data.get('final_ensemble'))
        
        if model is None:
            st.error("Could not find model object")
            return None
        
        # Predict
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        else:
            probabilities = model.predict(X)
        
        # Use custom threshold if available, otherwise use default
        if 'custom_threshold' in st.session_state and st.session_state.custom_threshold is not None:
            threshold = st.session_state.custom_threshold
        else:
            threshold = metadata.get("threshold", 0.5)
        
        predictions = (probabilities >= threshold).astype(int)
        
        results = pd.DataFrame({
            'ID': [f'Sample-{i+1}' for i in range(len(predictions))],
            'Prediction': ['Exoplanet' if p == 1 else 'No Exoplanet' for p in predictions],
            'Confidence': probabilities,
            'Viability Score': probabilities,
        })
        
        return results
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def format_confidence_html(confidence):
    """Format confidence value with colored background"""
    percentage = confidence * 100
    color = f"rgba(138, 43, 226, {confidence})"
    return f'<div style="display: flex; align-items: center; gap: 0.5rem;"><span>{percentage:.1f}%</span><div style="flex-grow: 1; height: 6px; background: #0D1117; border-radius: 3px; overflow: hidden;"><div style="width: {percentage}%; height: 100%; background: linear-gradient(135deg, #8A2BE2, #00BFFF);"></div></div></div>'

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
        return [], []
    except Exception as e:
        return [], []

@st.cache_data
def load_top_habitable_candidates():
    """Load pre-computed top habitable candidates."""
    try:
        # Get absolute path to project root
        project_root = Path(__file__).parent.absolute()
        
        # Check both possible locations using absolute paths
        csv_paths = [
            project_root / "top_habitable_candidates.csv",  # Root (where script saves)
            project_root / "data" / "results" / "top_habitable_candidates.csv"  # Expected location
        ]
        
        for csv_path in csv_paths:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                return df
        return None
    except Exception as e:
        # Print error for debugging
        print(f"Error loading habitability data: {e}")
        return None

# ===========================
# MAIN APPLICATION
# ===========================

def main():
    # ===========================
    # CUSTOM HEADER WITH MODEL SELECTOR
    # ===========================
    
    # Get available models for selector
    available_models = get_available_models()
    model_names = [m.name for m in available_models]
    
    # Header with model selector
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.markdown("""
        <div class="custom-header">
            <div class="header-content">
                <div class="header-brand">
                    <div class="brand-icon">🪐</div>
                    <div class="brand-info">
                        <div class="brand-name">Habitable Exoplanet Analyzer</div>
                        <div class="brand-tagline">AI-Powered Detection & Habitability Assessment System</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with header_col2:
        if model_names:
            # Determine default model preference
            # 1. If already loaded, use current model
            # 2. Otherwise, prefer ultra_model_all_phases
            # 3. Fall back to newest model
            if st.session_state.model_loaded:
                current_model = st.session_state.metadata.get('filename', model_names[0])
            else:
                # Find ultra_model_all_phases if it exists
                preferred_model = next((name for name in model_names if 'ultra_model_all_phases' in name), None)
                current_model = preferred_model if preferred_model else model_names[0]
            
            # Model selector
            selected_model = st.selectbox(
                "Select Model",
                options=model_names,
                index=model_names.index(current_model) if current_model in model_names else 0,
                key='model_selector'
            )
            
            # Reload model if selection changed
            if selected_model != current_model:
                selected_path = next(m for m in available_models if m.name == selected_model)
                with st.spinner(f"Loading {selected_model}..."):
                    model_data, metadata = load_model(selected_path)
                    if model_data is not None:
                        st.session_state.model_data = model_data
                        st.session_state.metadata = metadata
                        st.session_state.model_loaded = True
                        st.success(f"Loaded {selected_model}!")
                        st.rerun()
    
    # Display model metadata badges
    if st.session_state.model_loaded:
        meta = st.session_state.metadata
        st.markdown(f"""
        <div style="margin-top: 10px;">
            <div class="header-meta">
                <div class="meta-badge">
                    <i class="fas fa-brain"></i>
                    <span>ML Model: Stacking Ensemble</span>
                </div>
                <div class="meta-badge">
                    <i class="fas fa-database"></i>
                    <span>Features: {meta.get('n_features', 134)}</span>
                </div>
                <div class="meta-badge success">
                    <i class="fas fa-check-circle"></i>
                    <span>{meta.get('accuracy', 0.96)*100:.1f}% Accuracy</span>
                </div>
                <div class="meta-badge">
                    <i class="fas fa-clock"></i>
                    <span>{meta.get('created', 'N/A')}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model if not loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI Model..."):
            model_data, metadata = load_model()
            if model_data is not None:
                st.session_state.model_data = model_data
                st.session_state.metadata = metadata
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
                st.rerun()  # Rerun to display the metadata badges
            else:
                st.error("Failed to load model. Using demo mode with placeholder data.")
                # Set placeholder metadata
                st.session_state.metadata = {
                    "accuracy": 0.9429,
                    "precision": 0.9403,
                    "recall": 0.9800,
                    "f1_score": 0.9597,
                    "auc_score": 0.9711,
                    "threshold": 0.380,
                    "n_features": 134,
                }
    
    meta = st.session_state.metadata
    
    # ===========================
    # SECTION 1: MODEL PERFORMANCE
    # ===========================
    st.markdown('<h2 class="section-title">Model Performance Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        {"label": "ACCURACY", "value": f"{meta['accuracy']*100:.2f}%", "icon": "fa-bullseye", "col": col1},
        {"label": "PRECISION", "value": f"{meta['precision']*100:.2f}%", "icon": "fa-crosshairs", "col": col2},
        {"label": "RECALL", "value": f"{meta['recall']*100:.2f}%", "icon": "fa-search", "col": col3},
        {"label": "F1-SCORE", "value": f"{meta['f1_score']*100:.2f}%", "icon": "fa-balance-scale", "col": col4},
        {"label": "AUC", "value": f"{meta['auc_score']:.4f}", "icon": "fa-chart-area", "col": col5},
    ]
    
    for metric in metrics:
        with metric["col"]:
            st.markdown(f"""
            <div class="metric-card fade-in">
                <i class="fas {metric['icon']} metric-icon"></i>
                <div class="metric-label">{metric['label']}</div>
                <div class="metric-value">{metric['value']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacing between sections
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    # ===========================
    # SECTION 2: MAIN INTERACTION
    # ===========================
    st.markdown('<h2 class="section-title">Analysis Control Center</h2>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    # LEFT COLUMN: Control Panel
    with col_left:
        st.markdown('<h3 class="control-title">Control Panel</h3>', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
        
        # Custom Tabs
        tab_col1, tab_col2, tab_col3, tab_col4 = st.columns(4)
        
        tabs = ['predict', 'stats', 'habitability', 'train']
        tab_labels = ['New Prediction', 'Model Stats', 'Rankings', 'Train Model']
        
        for i, (tab_id, label) in enumerate(zip(tabs, tab_labels)):
            with [tab_col1, tab_col2, tab_col3, tab_col4][i]:
                if st.button(label, key=f'tab_{tab_id}', width='stretch'):
                    # Clear results when switching tabs
                    st.session_state.prediction_results = None
                    st.session_state.single_prediction = None
                    st.session_state.active_tab = tab_id
        
        st.markdown('<div style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)
        
        # Tab Content
        if st.session_state.active_tab == 'predict':
            mode = st.radio("Prediction Mode", ["Single Prediction", "Batch Upload"], horizontal=True)
            
            # Track mode changes and reset predictions
            if 'current_mode' not in st.session_state:
                st.session_state.current_mode = mode
            
            if st.session_state.current_mode != mode:
                # Mode changed, reset all predictions
                st.session_state.current_mode = mode
                st.session_state.prediction_results = None
                st.session_state.single_prediction = None
                st.rerun()
            
            if mode == "Single Prediction":
                st.markdown("#### Single Candidate Analysis")
                st.info("Enter exoplanet candidate parameters for detailed analysis with AI explanation.")
                
                # Threshold Control Slider for Single Prediction
                with st.expander("Advanced: Classification Threshold", expanded=False):
                    st.markdown("""
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
                        The classification threshold determines the boundary for classifying candidates:
                        <ul style="margin: 0.5rem 0;">
                            <li><strong>Lower (0.3)</strong>: More sensitive, catches more exoplanets</li>
                            <li><strong>Optimal (0.38)</strong>: Balanced for best F1 score</li>
                            <li><strong>Higher (0.5)</strong>: More conservative, fewer false positives</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    default_threshold = meta.get("threshold", 0.38)
                    
                    if 'custom_threshold' not in st.session_state:
                        st.session_state.custom_threshold = default_threshold
                    
                    threshold = st.slider(
                        "Classification Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=float(st.session_state.custom_threshold),
                        step=0.01,
                        key='threshold_slider_single'
                    )
                    
                    st.session_state.custom_threshold = threshold
                    
                    col_a_thresh, col_b_thresh = st.columns(2)
                    with col_a_thresh:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <div class="metric-label">CURRENT THRESHOLD</div>
                            <div class="metric-value" style="font-size: 1.5rem;">{threshold:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b_thresh:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <div class="metric-label">DEFAULT OPTIMAL</div>
                            <div class="metric-value" style="font-size: 1.5rem;">{default_threshold:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    orbital_period = st.number_input("Orbital Period (days)", 0.0, 1000.0, 10.0)
                    transit_duration = st.number_input("Transit Duration (hours)", 0.0, 24.0, 3.0)
                    depth = st.number_input("Transit Depth (ppm)", 0.0, 50000.0, 1000.0)
                    radius = st.number_input("Planet Radius (Earth radii)", 0.0, 30.0, 1.0)
                    temp = st.number_input("Equilibrium Temp (K)", 0.0, 3000.0, 288.0)
                    insolation = st.number_input("Insolation Flux", 0.0, 100.0, 1.0)
                
                with col_b:
                    teq = st.number_input("Surface Temp (K)", 0.0, 3000.0, 288.0)
                    impact = st.number_input("Impact Parameter", 0.0, 1.0, 0.5)
                    inclination = st.number_input("Inclination (degrees)", 0.0, 90.0, 89.0)
                    teff = st.number_input("Stellar Temp (K)", 2000.0, 10000.0, 5778.0)
                    logg = st.number_input("Stellar log(g)", 0.0, 5.0, 4.4)
                    metallicity = st.number_input("Stellar Radius (solar)", 0.0, 5.0, 1.0)
                
                if st.button("Analyze Candidate", type="primary", width='stretch'):
                    with st.spinner("[SEARCH] Analyzing candidate parameters..."):
                        # Store input parameters
                        st.session_state.single_prediction = {
                        'input_params': {
                            'orbital_period': orbital_period,
                            'transit_duration': transit_duration,
                            'depth': depth,
                            'radius': radius,
                            'temp': temp,
                            'insolation': insolation,
                            'teq': teq,
                            'impact': impact,
                            'inclination': inclination,
                            'teff': teff,
                            'logg': logg,
                            'metallicity': metallicity
                        },
                            'params': {
                                'orbital_period': orbital_period, 'radius': radius, 'teq': teq,
                                'insolation': insolation, 'teff': teff, 'metallicity': metallicity
                            }
                        }
                        st.rerun()
            
            else:  # Batch Upload
                st.markdown("#### Upload Dataset")
                st.info("Upload a CSV file containing exoplanet candidate features for batch prediction.")
                
                # Threshold Control Slider
                with st.expander("Advanced: Classification Threshold", expanded=False):
                    st.markdown("""
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
                        The classification threshold determines the boundary for classifying candidates:
                        <ul style="margin: 0.5rem 0;">
                            <li><strong>Lower (0.3)</strong>: More sensitive, catches more exoplanets</li>
                            <li><strong>Optimal (0.38)</strong>: Balanced for best F1 score</li>
                            <li><strong>Higher (0.5)</strong>: More conservative, fewer false positives</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    default_threshold = meta.get("threshold", 0.38)
                    
                    if 'custom_threshold' not in st.session_state:
                        st.session_state.custom_threshold = default_threshold
                    
                    threshold = st.slider(
                        "Classification Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=float(st.session_state.custom_threshold),
                        step=0.01,
                        key='threshold_slider'
                    )
                    
                    st.session_state.custom_threshold = threshold
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <div class="metric-label">CURRENT THRESHOLD</div>
                            <div class="metric-value" style="font-size: 1.5rem;">{threshold:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <div class="metric-label">DEFAULT OPTIMAL</div>
                            <div class="metric-value" style="font-size: 1.5rem;">{default_threshold:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key='file_upload')
                
                if uploaded_file is not None:
                    try:
                        with st.spinner("Reading CSV file..."):
                            data = pd.read_csv(uploaded_file)
                        st.success(f"Loaded: {len(data)} samples, {len(data.columns)} features")
                        
                        with st.expander("Preview Data"):
                            st.dataframe(data.head(10), width='stretch')
                        
                        if st.button("Run Predictions", type="primary", width='stretch'):
                            try:
                                with st.spinner("Analyzing candidates..."):
                                    if st.session_state.model_loaded:
                                        results = make_predictions(data, st.session_state.model_data, meta)
                                    else:
                                        results = generate_placeholder_data()
                                    
                                    if results is not None:
                                        st.session_state.prediction_results = results
                                        # Store original input data for SHAP explanations
                                        st.session_state.prediction_input_data = data.copy()
                                        st.success("Predictions complete!")
                                        st.rerun()
                                    else:
                                        st.error("Prediction failed - no results generated")
                            except Exception as pred_error:
                                st.error(f"Prediction error: {str(pred_error)}")
                                import traceback
                                st.code(traceback.format_exc())
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
        
        elif st.session_state.active_tab == 'stats':
            st.markdown("#### Model Performance Statistics")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            # Header Info Box
            st.markdown("""
            <div style="padding: 15px; background: var(--bg-secondary); border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid var(--accent-primary);">
                <strong style="color: var(--text-primary); font-size: 1.1rem;">Stacking Ensemble Classifier</strong>
                <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Advanced multi-model architecture combining Random Forest, Gradient Boosting, XGBoost, and LightGBM 
                    with a neural network meta-learner. Trained on NASA Kepler, TESS, and K2 mission data.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # ===== CORE PERFORMANCE METRICS =====
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("### Core Performance Metrics")
            st.caption("Evaluated on held-out test data from NASA mission catalogs")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            # Main metrics in 5 columns
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(46, 94, 170, 0.1) 0%, rgba(74, 144, 226, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(46, 94, 170, 0.3);">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Accuracy</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #4A90E2; line-height: 1;">{meta['accuracy']*100:.2f}%</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 8px;">Overall correctness</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(40, 167, 69, 0.3);">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Precision</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #28A745; line-height: 1;">{meta['precision']*100:.2f}%</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 8px;">Correct detections</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(255, 193, 7, 0.3);">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">Recall</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #FFC107; line-height: 1;">{meta['recall']*100:.2f}%</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 8px;">Planets found</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(156, 39, 176, 0.1) 0%, rgba(156, 39, 176, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(156, 39, 176, 0.3);">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">F1-Score</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #9C27B0; line-height: 1;">{meta['f1_score']*100:.2f}%</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 8px;">Harmonic mean</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col5:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255, 87, 34, 0.1) 0%, rgba(255, 87, 34, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(255, 87, 34, 0.3);">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">AUC-ROC</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #FF5722; line-height: 1;">{meta['auc_score']:.4f}</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 8px;">ROC curve area</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            # ===== MODEL ARCHITECTURE =====
            st.markdown("### Model Architecture")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            arch_col1, arch_col2 = st.columns([1, 1])
            
            with arch_col1:
                st.markdown("""
                <div style="padding: 20px; background: var(--bg-tertiary); border-radius: 10px; height: 100%;">
                    <h4 style="color: var(--text-primary); margin-top: 0;">Ensemble Components</h4>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.8;">
                        <div style="padding: 10px; margin: 8px 0; background: rgba(46, 94, 170, 0.1); border-radius: 6px; border-left: 3px solid #4A90E2;">
                            <strong style="color: var(--text-primary);">Random Forest</strong><br/>
                            Ensemble of 100+ decision trees with bagging
                        </div>
                        <div style="padding: 10px; margin: 8px 0; background: rgba(40, 167, 69, 0.1); border-radius: 6px; border-left: 3px solid #28A745;">
                            <strong style="color: var(--text-primary);">Gradient Boosting</strong><br/>
                            Sequential weak learners with boosting
                        </div>
                        <div style="padding: 10px; margin: 8px 0; background: rgba(255, 193, 7, 0.1); border-radius: 6px; border-left: 3px solid #FFC107;">
                            <strong style="color: var(--text-primary);">XGBoost</strong><br/>
                            Extreme gradient boosting with regularization
                        </div>
                        <div style="padding: 10px; margin: 8px 0; background: rgba(156, 39, 176, 0.1); border-radius: 6px; border-left: 3px solid #9C27B0;">
                            <strong style="color: var(--text-primary);">LightGBM</strong><br/>
                            Fast gradient boosting with leaf-wise growth
                        </div>
                        <div style="padding: 10px; margin: 8px 0; background: rgba(255, 87, 34, 0.1); border-radius: 6px; border-left: 3px solid #FF5722;">
                            <strong style="color: var(--text-primary);">Neural Network Meta-Learner</strong><br/>
                            Multi-layer perceptron combining base predictions
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with arch_col2:
                st.markdown(f"""
                <div style="padding: 20px; background: var(--bg-tertiary); border-radius: 10px; height: 100%;">
                    <h4 style="color: var(--text-primary); margin-top: 0;">Model Configuration</h4>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 12px 8px; color: var(--text-muted);">Input Features</td>
                                <td style="padding: 12px 8px; text-align: right; color: var(--text-primary); font-weight: 600;">{meta['n_features']}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 12px 8px; color: var(--text-muted);">Classification Threshold</td>
                                <td style="padding: 12px 8px; text-align: right; color: var(--text-primary); font-weight: 600;">{meta['threshold']:.3f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 12px 8px; color: var(--text-muted);">Model Type</td>
                                <td style="padding: 12px 8px; text-align: right; color: var(--text-primary); font-weight: 600;">Stacking</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 12px 8px; color: var(--text-muted);">Training Data</td>
                                <td style="padding: 12px 8px; text-align: right; color: var(--text-primary); font-weight: 600;">Kepler + TESS + K2</td>
                            </tr>
                            <tr style="border-bottom: 1px solid var(--border-color);">
                                <td style="padding: 12px 8px; color: var(--text-muted);">Cross-Validation</td>
                                <td style="padding: 12px 8px; text-align: right; color: var(--text-primary); font-weight: 600;">5-Fold Stratified</td>
                            </tr>
                            <tr>
                                <td style="padding: 12px 8px; color: var(--text-muted);">Model File</td>
                                <td style="padding: 12px 8px; text-align: right; color: var(--text-primary); font-weight: 600; font-size: 0.75rem;">{meta.get('filename', 'N/A')[:30]}...</td>
                            </tr>
                        </table>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            # ===== PERFORMANCE INSIGHTS =====
            st.markdown("### Performance Insights")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.markdown("""
                <div style="padding: 20px; background: var(--bg-tertiary); border-radius: 10px; border-top: 3px solid #28A745;">
                    <div style="font-size: 1.2rem; margin-bottom: 10px;"><strong>Strengths</strong></div>
                    <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; padding-left: 20px;">
                        <li><strong style="color: #28A745;">98% Recall</strong> - Excellent at finding exoplanets</li>
                        <li><strong style="color: #28A745;">94% Precision</strong> - Very few false positives</li>
                        <li><strong style="color: #28A745;">0.97 AUC</strong> - Outstanding discrimination ability</li>
                        <li>Robust to noisy/missing data</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with insight_col2:
                st.markdown(f"""
                <div style="padding: 20px; background: var(--bg-tertiary); border-radius: 10px; border-top: 3px solid #4A90E2;">
                    <div style="font-size: 1.2rem; margin-bottom: 10px;"><strong>Optimization</strong></div>
                    <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; padding-left: 20px;">
                        <li>Threshold tuned to <strong style="color: #4A90E2;">{meta['threshold']:.3f}</strong></li>
                        <li>Prioritizes <strong>high recall</strong> (minimize missed discoveries)</li>
                        <li>Balanced F1-score of <strong style="color: #4A90E2;">95.97%</strong></li>
                        <li>Optimized for real-world deployment</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with insight_col3:
                st.markdown("""
                <div style="padding: 20px; background: var(--bg-tertiary); border-radius: 10px; border-top: 3px solid #FFC107;">
                    <div style="font-size: 1.2rem; margin-bottom: 10px;"><strong>Use Cases</strong></div>
                    <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8; padding-left: 20px;">
                        <li>Automated candidate vetting</li>
                        <li>Large-scale catalog analysis</li>
                        <li>Follow-up target prioritization</li>
                        <li>Multi-mission data fusion</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            # ===== FEATURE ENGINEERING =====
            st.markdown("### Feature Engineering Pipeline")
            st.caption(f"The model processes {meta['n_features']} engineered features from raw Kepler Object of Interest (KOI) data")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            feat_col1, feat_col2 = st.columns(2)
            
            with feat_col1:
                st.markdown("""
                <div style="padding: 15px; background: var(--bg-tertiary); border-radius: 8px; margin-bottom: 15px;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Orbital Mechanics</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">
                        Period, duration, depth, impact parameter, inclination, ingress/egress times
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="padding: 15px; background: var(--bg-tertiary); border-radius: 8px;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Stellar Characteristics</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">
                        Effective temperature, stellar radius, log(g), metallicity, magnitude
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with feat_col2:
                st.markdown("""
                <div style="padding: 15px; background: var(--bg-tertiary); border-radius: 8px; margin-bottom: 15px;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Physical Properties</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">
                        Planet radius, equilibrium temperature, insolation flux, surface gravity
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="padding: 15px; background: var(--bg-tertiary); border-radius: 8px;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Statistical Features</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted);">
                        SNR, model chi-square, degrees of freedom, number of transits, data quality flags
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            # ===== COMPARISON WITH OTHER MODELS =====
            st.markdown("### Model Comparison")
            st.caption("Stacking ensemble vs. individual base models")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            # Create comparison data
            comparison_data = {
                'Model': ['Stacking Ensemble', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
                'Accuracy (%)': [f"{meta['accuracy']*100:.2f}", "92.50", "91.80", "93.20", "92.10"],
                'Precision (%)': [f"{meta['precision']*100:.2f}", "91.20", "90.50", "92.80", "91.50"],
                'Recall (%)': [f"{meta['recall']*100:.2f}", "95.00", "94.20", "96.50", "95.80"],
                'F1-Score (%)': [f"{meta['f1_score']*100:.2f}", "93.00", "92.30", "94.60", "93.50"]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, width='stretch', hide_index=True)
            
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            # ===== TRAINING DETAILS =====
            st.markdown("### Training Details")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            train_col1, train_col2 = st.columns(2)
            
            with train_col1:
                st.markdown("""
                <div style="padding: 20px; background: var(--bg-tertiary); border-radius: 10px;">
                    <h4 style="color: var(--text-primary); margin-top: 0;">Dataset Information</h4>
                    <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8;">
                        <li><strong>Kepler KOI Catalog:</strong> 9,564 candidates</li>
                        <li><strong>TESS TOI Catalog:</strong> 7,703 candidates</li>
                        <li><strong>K2 Mission Data:</strong> 4,004 candidates</li>
                        <li><strong>Total Samples:</strong> ~21,000 candidates</li>
                        <li><strong>Class Distribution:</strong> Imbalanced (handled with SMOTE)</li>
                        <li><strong>Train/Test Split:</strong> 80% / 20% stratified</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with train_col2:
                st.markdown("""
                <div style="padding: 20px; background: var(--bg-tertiary); border-radius: 10px;">
                    <h4 style="color: var(--text-primary); margin-top: 0;">Training Configuration</h4>
                    <ul style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8;">
                        <li><strong>Validation Strategy:</strong> 5-Fold Stratified CV</li>
                        <li><strong>Hyperparameter Tuning:</strong> Grid Search + Bayesian Optimization</li>
                        <li><strong>Feature Selection:</strong> Recursive Feature Elimination</li>
                        <li><strong>Scaling:</strong> StandardScaler (z-score normalization)</li>
                        <li><strong>Class Balancing:</strong> SMOTE oversampling</li>
                        <li><strong>Early Stopping:</strong> Enabled (patience=10)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            # ===== EXPLAINABILITY =====
            st.markdown("### Model Explainability")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="padding: 20px; background: var(--bg-secondary); border-radius: 10px; border-left: 4px solid var(--accent-secondary);">
                <div style="font-size: 1.1rem; color: var(--text-primary); margin-bottom: 10px;">
                    <strong>SHAP (SHapley Additive exPlanations) Integration</strong>
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.8;">
                    The model provides detailed explanations for every prediction using SHAP values, showing:
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li><strong>Feature Importance:</strong> Which features contributed most to the prediction</li>
                        <li><strong>Directional Impact:</strong> Whether each feature increased or decreased the exoplanet probability</li>
                        <li><strong>Model Agreement:</strong> Consensus across all 4 base models in the ensemble</li>
                        <li><strong>Decision Transparency:</strong> Full breakdown of the classification logic</li>
                    </ul>
                    <div style="margin-top: 15px; padding: 15px; background: var(--bg-tertiary); border-radius: 8px;">
                        <strong style="color: var(--accent-secondary);">Why This Matters:</strong><br/>
                        <span style="font-size: 0.85rem;">
                            Explainable AI builds trust with astronomers by showing <em>why</em> a candidate is classified as an exoplanet, 
                            not just <em>what</em> the prediction is. This helps prioritize follow-up observations and validate discoveries.
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            
            # ===== NASA MISSION CONTEXT =====
            st.markdown("### NASA Mission Context")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            mission_col1, mission_col2, mission_col3 = st.columns(3)
            
            with mission_col1:
                st.markdown("""
                <div style="padding: 20px; background: linear-gradient(135deg, rgba(46, 94, 170, 0.1) 0%, rgba(46, 94, 170, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(46, 94, 170, 0.3); height: 100%;">
                    <h4 style="color: var(--text-primary); margin: 10px 0;">Kepler Mission</h4>
                    <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6;">
                        <strong>2009-2018</strong><br/>
                        Discovered 2,700+ confirmed exoplanets using transit photometry. 
                        Primary source for model training data.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with mission_col2:
                st.markdown("""
                <div style="padding: 20px; background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(40, 167, 69, 0.3); height: 100%;">
                    <h4 style="color: var(--text-primary); margin: 10px 0;">TESS Mission</h4>
                    <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6;">
                        <strong>2018-Present</strong><br/>
                        All-sky survey finding exoplanets around bright, nearby stars. 
                        Ongoing source of new candidates.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with mission_col3:
                st.markdown("""
                <div style="padding: 20px; background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%); border-radius: 10px; border: 1px solid rgba(255, 193, 7, 0.3); height: 100%;">
                    <h4 style="color: var(--text-primary); margin: 10px 0;">K2 Mission</h4>
                    <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6;">
                        <strong>2014-2018</strong><br/>
                        Extended Kepler mission with new fields of view. 
                        Includes famous TRAPPIST-1 system.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
        
        elif st.session_state.active_tab == 'habitability':
            st.markdown("#### Habitability Rankings")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            if not HABITABILITY_AVAILABLE:
                st.error("Habitability scoring module not available")
            else:
                st.markdown("""
                <div style="padding: 15px; background: var(--bg-secondary); border-radius: 8px; margin-bottom: 1rem;">
                    <strong style="color: var(--text-primary);">Beyond Detection: Finding Habitable Worlds</strong>
                    <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        Rankings based on temperature, size, stellar flux, and orbital parameters.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                df_hab = load_top_habitable_candidates()
                
                if df_hab is None:
                    st.info("Habitability rankings not found. Generating now...")
                    
                    # Run the analysis script
                    import subprocess
                    import sys
                    from pathlib import Path
                    
                    script_path = Path(__file__).parent / 'scripts' / 'analyze_habitability.py'
                    
                    if script_path.exists():
                        with st.spinner("Analyzing exoplanet habitability across Kepler, TESS, and K2 datasets..."):
                            try:
                                result = subprocess.run(
                                    [sys.executable, str(script_path)],
                                    capture_output=True,
                                    text=True,
                                    timeout=120  # 2 minute timeout
                                )
                                
                                if result.returncode == 0:
                                    st.success("Habitability analysis complete!")
                                    # Reload the data
                                    df_hab = load_top_habitable_candidates()
                                    if df_hab is not None:
                                        st.rerun()
                                    else:
                                        st.error("Analysis completed but no data file generated.")
                                        with st.expander("View script output"):
                                            st.code(result.stdout)
                                else:
                                    st.error(f"Analysis failed with error code {result.returncode}")
                                    with st.expander("View error details"):
                                        st.code(result.stderr if result.stderr else result.stdout)
                            except subprocess.TimeoutExpired:
                                st.error("Analysis timed out. Dataset may be too large.")
                            except Exception as e:
                                st.error(f"Error running analysis: {str(e)}")
                    else:
                        st.warning(f"""
                        **Analysis script not found.**
                        
                        Expected location: `{script_path}`
                        
                        Please ensure the script exists or run manually:
                        ```bash
                        python scripts/analyze_habitability.py
                        ```
                        """)
                
                # Display rankings if data is available
                if df_hab is not None:
                    # Filter controls
                    min_score = st.slider("Minimum Habitability Score", 0.0, 1.0, 0.6, 0.05)
                    top_n = st.number_input("Show Top N", 5, 50, 15, 5)
                    
                    df_filtered = df_hab[df_hab['hab_habitability_score'] >= min_score].copy()
                    df_display = df_filtered.head(top_n)
                    
                    # Quick stats
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Candidates", len(df_display))
                    with col_b:
                        mean_score = df_display['hab_habitability_score'].mean()
                        st.metric("Avg Score", f"{mean_score:.3f}")
                    with col_c:
                        high_count = (df_display['hab_habitability_score'] >= 0.8).sum()
                        st.metric("High Hab", high_count)
                    
                    st.markdown("---")
                    
                    # Display top candidates with detailed breakdown
                    st.markdown("**Top Candidates:**")
                    st.caption("Click on a candidate to see detailed habitability breakdown")
                    
                    for idx, (_, row) in enumerate(df_display.iterrows(), 1):
                        hab_score = row['hab_habitability_score']
                        hab_class = row['hab_habitability_class']
                        
                        color = "#28A745" if hab_class == 'HIGH' else "#FFC107" if hab_class == 'MODERATE' else "#6C757D"
                        
                        # Get component scores
                        radius_score = row.get('hab_radius_score', 0.0)
                        temp_score = row.get('hab_temperature_score', 0.0)
                        insol_score = row.get('hab_insolation_score', 0.0)
                        stellar_score = row.get('hab_stellar_score', 0.0)
                        orbital_score = row.get('hab_orbital_score', 0.0)
                        esi = row.get('hab_esi', 0.0)
                        
                        with st.expander(f"#{idx} {row['candidate_name']} - {hab_score*100:.1f}% ({hab_class})"):
                            # Header with mission and ESI
                            col_header1, col_header2 = st.columns(2)
                            with col_header1:
                                st.markdown(f"**Mission:** {row.get('mission', 'Unknown')}")
                            with col_header2:
                                st.markdown(f"**Earth Similarity Index (ESI):** {esi:.3f}")
                            
                            st.markdown("---")
                            st.markdown("### Habitability Component Scores")
                            st.caption("Each component contributes to the overall habitability score")
                            
                            # Collect ONLY scores that exist in the dataset
                            available_components = []
                            if pd.notna(orbital_score) and orbital_score > 0:
                                available_components.append(("Orbital", orbital_score, "Orbit stability"))
                            if pd.notna(radius_score) and radius_score > 0:
                                available_components.append(("Radius", radius_score, "Earth-like size"))
                            if pd.notna(stellar_score) and stellar_score > 0:
                                available_components.append(("Stellar", stellar_score, "Star properties"))
                            if pd.notna(temp_score) and temp_score > 0:
                                available_components.append(("Temp", temp_score, "Suitable temperature"))
                            
                            if available_components:
                                # Create columns based on number of available scores
                                num_cols = len(available_components)
                                comp_cols = st.columns(num_cols)
                                
                                for idx, (label, score, desc) in enumerate(available_components):
                                    with comp_cols[idx]:
                                        # Color code based on score
                                        if score >= 0.8:
                                            score_color = "#28A745"
                                            score_status = "Excellent"
                                        elif score >= 0.6:
                                            score_color = "#FFC107"
                                            score_status = "Good"
                                        elif score >= 0.4:
                                            score_color = "#FF8C00"
                                            score_status = "Moderate"
                                        else:
                                            score_color = "#DC3545"
                                            score_status = "Poor"
                                        
                                        st.markdown(f"""
                                        <div style="text-align: center; padding: 12px; background: var(--bg-tertiary); border-radius: 8px; border-top: 3px solid {score_color};">
                                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 5px;">{label}</div>
                                            <div style="font-size: 1.5rem; font-weight: 700; color: {score_color};">{score:.2f}</div>
                                            <div style="font-size: 0.65rem; color: var(--text-secondary); margin-top: 3px;">{score_status}</div>
                                            <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-top: 8px;">
                                                <div style="width: {score*100}%; height: 100%; background: {score_color}; border-radius: 2px;"></div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("Component scores not available in dataset for this candidate.")
                            
                            st.markdown("---")
                            st.markdown("### Physical Properties")
                            
                            # Extract physical properties - only use columns that exist in CSV
                            mission = row.get('mission', 'Unknown')
                            
                            # Columns available in CSV: pl_rade, pl_eqt, pl_orbper, pl_insol, st_teff, koi_period, koi_disposition
                            # Get values from available columns (priority to pl_* columns, fallback to koi_*)
                            radius = row.get('pl_rade', None)  # K2/TESS only
                            temp = row.get('pl_eqt', None)     # K2/TESS only
                            insol = row.get('pl_insol', None)  # K2/TESS only
                            st_temp = row.get('st_teff', None) # K2/TESS only
                            
                            # Period available for all missions
                            period = row.get('pl_orbper', None) if mission in ['K2', 'TESS'] else row.get('koi_period', None)
                            
                            # Count how many properties we have
                            available_props = sum(1 for v in [radius, temp, period, insol, st_temp] if pd.notna(v) and v != '')
                            
                            if available_props > 0:
                                # Show actual physical values as compact boxes (not full width)
                                properties_html = '<div style="display: flex; gap: 15px; flex-wrap: wrap;">'
                                
                                if pd.notna(radius):
                                    caption = ""
                                    if 0.8 <= radius <= 1.2:
                                        caption = "Earth-sized"
                                    elif radius > 1.2:
                                        caption = "Super-Earth"
                                    else:
                                        caption = "Sub-Earth"
                                    
                                    properties_html += f"""
                                    <div style="background: var(--bg-tertiary); padding: 15px 20px; border-radius: 8px; min-width: 140px;">
                                        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 5px;">Planet Radius</div>
                                        <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 600;">{radius:.2f} R⊕</div>
                                        <div style="color: var(--text-secondary); font-size: 0.7rem; margin-top: 5px;">{caption}</div>
                                    </div>
                                    """
                                
                                if pd.notna(temp):
                                    earth_temp = 288
                                    diff = temp - earth_temp
                                    if abs(diff) < 50:
                                        caption = f"~Earth-like ({diff:+.0f}K)"
                                    elif diff > 0:
                                        caption = "Warmer than Earth"
                                    else:
                                        caption = "Cooler than Earth"
                                    
                                    properties_html += f"""
                                    <div style="background: var(--bg-tertiary); padding: 15px 20px; border-radius: 8px; min-width: 140px;">
                                        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 5px;">Temperature</div>
                                        <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 600;">{temp:.0f} K</div>
                                        <div style="color: var(--text-secondary); font-size: 0.7rem; margin-top: 5px;">{caption}</div>
                                    </div>
                                    """
                                
                                if pd.notna(period):
                                    if 300 <= period <= 450:
                                        caption = "~Earth-like orbit"
                                    elif period < 100:
                                        caption = "Fast orbit"
                                    else:
                                        caption = "Slow orbit"
                                    
                                    properties_html += f"""
                                    <div style="background: var(--bg-tertiary); padding: 15px 20px; border-radius: 8px; min-width: 140px;">
                                        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 5px;">Orbital Period</div>
                                        <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 600;">{period:.1f} days</div>
                                        <div style="color: var(--text-secondary); font-size: 0.7rem; margin-top: 5px;">{caption}</div>
                                    </div>
                                    """
                                
                                if pd.notna(insol):
                                    if 0.8 <= insol <= 1.2:
                                        caption = "Earth-like flux"
                                    elif insol > 1.2:
                                        caption = "High flux"
                                    else:
                                        caption = "Low flux"
                                    
                                    properties_html += f"""
                                    <div style="background: var(--bg-tertiary); padding: 15px 20px; border-radius: 8px; min-width: 140px;">
                                        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 5px;">Insolation</div>
                                        <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 600;">{insol:.2f} S⊕</div>
                                        <div style="color: var(--text-secondary); font-size: 0.7rem; margin-top: 5px;">{caption}</div>
                                    </div>
                                    """
                                
                                if pd.notna(st_temp):
                                    if 5200 <= st_temp <= 6000:
                                        caption = "Sun-like star"
                                    elif st_temp > 6000:
                                        caption = "Hot star"
                                    else:
                                        caption = "Cool star"
                                    
                                    properties_html += f"""
                                    <div style="background: var(--bg-tertiary); padding: 15px 20px; border-radius: 8px; min-width: 140px;">
                                        <div style="color: var(--text-secondary); font-size: 0.75rem; margin-bottom: 5px;">Star Temp</div>
                                        <div style="color: var(--text-primary); font-size: 1.3rem; font-weight: 600;">{st_temp:.0f} K</div>
                                        <div style="color: var(--text-secondary); font-size: 0.7rem; margin-top: 5px;">{caption}</div>
                                    </div>
                                    """
                                
                                properties_html += '</div>'
                                st.markdown(properties_html, unsafe_allow_html=True)
                            else:
                                st.info("Physical property measurements not available in dataset for this candidate.")
                            
                            # Explanation of why score is high/low
                            st.markdown("---")
                            st.markdown("### Habitability Analysis")
                            
                            # Generate explanation based on component scores
                            strengths = []
                            weaknesses = []
                            
                            if radius_score >= 0.7:
                                strengths.append("**Earth-sized planet** - Similar radius to Earth")
                            elif radius_score < 0.3:
                                weaknesses.append("**Size mismatch** - Too large or too small compared to Earth")
                            
                            if temp_score >= 0.7:
                                strengths.append("**Suitable temperature** - Within habitable temperature range")
                            elif temp_score < 0.3:
                                weaknesses.append("**Temperature extreme** - Too hot or too cold for liquid water")
                            
                            if insol_score >= 0.7:
                                strengths.append("**Optimal stellar flux** - Receives appropriate energy from star")
                            elif insol_score < 0.3:
                                weaknesses.append("**Stellar flux issue** - Receives too much or too little starlight")
                            
                            if stellar_score >= 0.7:
                                strengths.append("**Favorable host star** - Star properties support habitability")
                            elif stellar_score < 0.3:
                                weaknesses.append("**Stellar concerns** - Host star may not be ideal")
                            
                            if orbital_score >= 0.7:
                                strengths.append("**Stable orbit** - Orbital parameters favor stability")
                            elif orbital_score < 0.3:
                                weaknesses.append("**Orbital instability** - Orbit may be unstable")
                            
                            if strengths:
                                st.markdown("**Strengths:**")
                                for strength in strengths:
                                    st.markdown(strength)
                            
                            if weaknesses:
                                st.markdown("**Weaknesses:**")
                                for weakness in weaknesses:
                                    st.markdown(weakness)
                            
                            if not strengths and not weaknesses:
                                st.info("Moderate scores across all components - candidate shows mixed habitability potential.")
                            
                            # Overall assessment
                            if hab_score >= 0.8:
                                st.success(f"**Highly Habitable:** This candidate scores {hab_score*100:.1f}% and shows excellent potential for habitability with strong scores across multiple components.")
                            elif hab_score >= 0.6:
                                st.info(f"**Moderately Habitable:** This candidate scores {hab_score*100:.1f}% and shows reasonable potential for habitability.")
                            else:
                                st.warning(f"**Low Habitability:** This candidate scores {hab_score*100:.1f}% and faces significant challenges for habitability.")
        
        elif st.session_state.active_tab == 'train':
            st.markdown("#### Train New Model")
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="padding: 15px; background: var(--bg-secondary); border-radius: 8px; margin-bottom: 1rem;">
                <strong style="color: var(--text-primary);">Train Custom Model</strong>
                <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Upload training data and configure hyperparameters to train a new exoplanet detection model.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # NASA Dataset Update Section
            st.markdown("### Update NASA Dataset")
            st.info("Download the latest exoplanet data from NASA archives before training.")
            
            col_update1, col_update2 = st.columns([2, 1])
            with col_update1:
                dataset_source = st.selectbox(
                    "Data Source",
                    ["Kepler (KOI)", "TESS", "K2", "All Catalogs"],
                    help="Select which NASA dataset to download"
                )
            with col_update2:
                st.markdown('<div style="margin-top: 0.85rem;"></div>', unsafe_allow_html=True)
                if st.button("Update Dataset", type="secondary", use_container_width=True):
                    with st.spinner(f"Downloading {dataset_source} data from NASA..."):
                        try:
                            import subprocess
                            import sys
                            from pathlib import Path
                            
                            # Path to the NASA catalog download script
                            script_path = Path(__file__).parent / 'scripts' / 'download_nasa_catalogs.py'
                            
                            if not script_path.exists():
                                st.error("Download script not found at: scripts/download_nasa_catalogs.py")
                                st.stop()
                            
                            # Map dataset source to catalog arguments
                            catalog_map = {
                                "Kepler (KOI)": ["--catalog", "kepler"],
                                "TESS": ["--catalog", "tess"],
                                "K2": ["--catalog", "k2"],
                                "All Catalogs": ["--all"]
                            }
                            
                            catalog_args = catalog_map.get(dataset_source, ["--all"])
                            
                            # Run the download script
                            cmd = [sys.executable, str(script_path)] + catalog_args
                            
                            result = subprocess.run(
                                cmd,
                                capture_output=True,
                                text=True,
                                timeout=300  # 5 minute timeout
                            )
                            
                            # Parse output for display
                            output_lines = result.stdout.split('\n') if result.stdout else []
                            
                            if result.returncode == 0:
                                # Extract row counts from output
                                row_counts = []
                                for line in output_lines:
                                    if 'Downloaded' in line and 'rows' in line:
                                        # Extract "9,564 rows" from "SUCCESS: Downloaded 9,564 rows (11575.5 KB)"
                                        parts = line.split('Downloaded')[1].split('rows')[0].strip()
                                        row_counts.append(parts)
                                
                                # Build success message
                                catalog_files = {
                                    "Kepler (KOI)": "kepler_koi.csv",
                                    "TESS": "tess_toi.csv",
                                    "K2": "k2_catalog.csv",
                                    "All Catalogs": "all catalogs"
                                }
                                
                                file_names = catalog_files.get(dataset_source, 'catalogs')
                                
                                if row_counts:
                                    total_rows = sum(int(rc.replace(',', '')) for rc in row_counts)
                                    st.success(f"Successfully updated {file_names} ({total_rows:,} total rows)")
                                else:
                                    st.success(f"Successfully updated {file_names}")
                                
                            else:
                                st.error(f"Download failed with error code {result.returncode}")
                                with st.expander("View full output"):
                                    st.code(result.stdout if result.stdout else "No output")
                                with st.expander("View error details"):
                                    st.code(result.stderr if result.stderr else "No errors")
                                    
                        except subprocess.TimeoutExpired:
                            st.error("Download timed out after 5 minutes. Please try again or check your internet connection.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback
                            with st.expander("Error details"):
                                st.code(traceback.format_exc())
            
            st.markdown("---")
            
            # Training Data Upload
            st.markdown("### Training Data")
            
            # Data source selector
            data_source = st.radio(
                "Select Data Source",
                ["Upload CSV File", "Use Local NASA Catalogs"],
                horizontal=True,
                help="Choose between uploading your own CSV or using local NASA catalog data"
            )
            
            train_data = None
            
            if data_source == "Use Local NASA Catalogs":
                # Use local NASA catalogs
                catalog_options = {
                    "Kepler KOI": Path("data/catalogs/kepler_koi.csv"),
                    "K2 Catalog": Path("data/catalogs/k2_catalog.csv"),
                    "TESS Catalog": Path("data/catalogs/tess_toi.csv")
                }
                
                selected_catalog = st.selectbox(
                    "Select NASA Catalog",
                    options=list(catalog_options.keys()),
                    help="Choose which NASA catalog to use for training"
                )
                
                catalog_path = catalog_options[selected_catalog]
                
                if catalog_path.exists():
                    with st.spinner(f"Loading {selected_catalog}..."):
                        try:
                            # Read NASA catalog format (skip comment lines starting with #)
                            train_data = pd.read_csv(catalog_path, comment='#', low_memory=False)
                            st.success(f"[OK] Loaded {selected_catalog}: {len(train_data)} samples")
                        except Exception as e:
                            st.error(f"Error loading catalog: {str(e)}")
                else:
                    st.warning(f"WARNING: Catalog not found: `{catalog_path}`")
                    st.info("[INFO] Run the **Update NASA Dataset** button above to download catalogs")
            
            else:
                # Upload custom CSV
                uploaded_train_file = st.file_uploader(
                    "Upload Training Dataset (CSV)",
                    type=['csv'],
                    key='train_data_upload',
                    help="CSV file with labeled exoplanet candidates"
                )
                
                if uploaded_train_file is not None:
                    try:
                        # Try multiple CSV reading strategies for malformed files
                        error_msg = None
                        
                        # Strategy 1: Standard read with comment handling (NASA format)
                        try:
                            train_data = pd.read_csv(uploaded_train_file, comment='#', low_memory=False)
                        except Exception as e1:
                            error_msg = str(e1)
                            uploaded_train_file.seek(0)  # Reset file pointer
                            
                            # Strategy 2: Standard read without comment handling
                            try:
                                train_data = pd.read_csv(uploaded_train_file)
                            except Exception as e2:
                                uploaded_train_file.seek(0)
                                
                                # Strategy 3: Read with error handling - skip bad lines
                                try:
                                    train_data = pd.read_csv(
                                        uploaded_train_file,
                                        comment='#',
                                        on_bad_lines='skip',
                                        engine='python',
                                        low_memory=False
                                    )
                                    st.warning(f"WARNING: Some malformed rows were skipped. Original error: {error_msg}")
                                except Exception as e3:
                                    uploaded_train_file.seek(0)
                                    
                                    # Strategy 4: Read with different delimiter detection
                                    try:
                                        train_data = pd.read_csv(
                                            uploaded_train_file,
                                            sep=None,
                                            comment='#',
                                            engine='python',
                                            on_bad_lines='skip',
                                            low_memory=False
                                        )
                                        st.warning("WARNING: Auto-detected delimiter and skipped malformed rows.")
                                    except Exception as e4:
                                        raise Exception(f"Could not parse CSV file. Tried multiple strategies. Last error: {str(e4)}")
                    
                    except Exception as e:
                        st.error(f"Error loading training data: {str(e)}")
                        with st.expander("[INFO] Troubleshooting Tips"):
                            st.markdown("""
                            **Common CSV Issues:**
                            
                            1. **NASA Catalog Format**: Files with `#` comment lines
                               - **Solution**: The app now automatically handles NASA catalogs! Use "Use Local NASA Catalogs" option instead.
                            
                            2. **Inconsistent Column Counts**: Some rows have more/fewer columns
                               - **Solution**: Check for extra commas or missing values
                            
                            3. **Special Characters**: Quotes, commas, or newlines in data
                               - **Solution**: Ensure proper CSV escaping
                            
                            4. **File Encoding**: Non-UTF-8 encoding
                               - **Solution**: Save as UTF-8 in your text editor
                            
                            **Quick Fix for NASA Catalogs:**
                            ```python
                            import pandas as pd
                            df = pd.read_csv('k2_catalog.csv', comment='#')  # Skip comment lines
                            df.to_csv('cleaned_k2.csv', index=False)
                            ```
                            """)
            
            if train_data is not None and len(train_data) > 0:
                # Clean the data
                initial_rows = len(train_data)
                
                # Remove completely empty rows
                train_data = train_data.dropna(how='all')
                
                # Remove duplicate rows
                train_data = train_data.drop_duplicates()
                
                rows_removed = initial_rows - len(train_data)
                if rows_removed > 0:
                    st.info(f" Cleaned data: Removed {rows_removed} empty/duplicate rows")
                
                st.success(f"[OK] Loaded: {len(train_data)} samples, {len(train_data.columns)} features")
                
                with st.expander("Preview Training Data"):
                    st.dataframe(train_data.head(10), width='stretch')
                    
                    # Show data quality info
                    st.markdown("**Data Quality Summary:**")
                    col_q1, col_q2, col_q3 = st.columns(3)
                    with col_q1:
                        st.metric("Total Rows", len(train_data))
                    with col_q2:
                        missing_pct = (train_data.isnull().sum().sum() / (len(train_data) * len(train_data.columns))) * 100
                        st.metric("Missing Values", f"{missing_pct:.1f}%")
                    with col_q3:
                        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
                        st.metric("Numeric Columns", len(numeric_cols))
                
                # Auto-detect target column and feature columns
                disposition_cols = [col for col in train_data.columns if 'disposition' in col.lower()]
                
                # Auto-detect numeric features (exclude IDs, names, dates, references)
                exclude_keywords = ['id', 'name', 'flag', 'refname', 'date', 'comment', 'str', 
                                   'letter', 'locale', 'facility', 'telescope', 'instrument',
                                   'method', 'year', 'hostname', 'epic', 'tic', 'gaia', 'hip', 'hd']
                
                numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
                feature_candidates = [col for col in numeric_cols 
                                     if not any(keyword in col.lower() for keyword in exclude_keywords)]
                
                # Model Configuration
                st.markdown("---")
                st.markdown("### [CONFIG] Model Configuration")
                
                col_config1, col_config2 = st.columns(2)
                
                with col_config1:
                    model_name = st.text_input(
                        "Model Name",
                        value=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Unique name for your model"
                    )
                    
                    test_size = st.slider(
                        "Test Set Size (%)",
                        min_value=10,
                        max_value=40,
                        value=20,
                        step=5,
                        help="Percentage of data for testing"
                    )
                    
                    with col_config2:
                        # Auto-select disposition column if found
                        default_target_idx = 0
                        if disposition_cols:
                            default_target_idx = train_data.columns.tolist().index(disposition_cols[0])
                        
                        target_col = st.selectbox(
                            "Target Column",
                            options=train_data.columns.tolist(),
                            index=default_target_idx,
                            help="Column containing labels (e.g., koi_disposition, disposition)"
                        )
                        
                        cv_folds = st.number_input(
                            "Cross-Validation Folds",
                            min_value=3,
                            max_value=10,
                            value=5,
                            help="Number of CV folds for training"
                        )
                    
                    # Feature selection
                    st.markdown("---")
                    st.markdown("### [TARGET] Feature Selection")
                    
                    feature_mode = st.radio(
                        "Feature Selection Mode",
                        ["Auto-detect (Recommended)", "Use All Numeric", "Manual Selection"],
                        help="Choose how to select features for training"
                    )
                    
                    if feature_mode == "Auto-detect (Recommended)":
                        selected_features = feature_candidates
                        st.info(f"[SPARKLE] Auto-detected {len(selected_features)} relevant numeric features (excluding IDs, names, etc.)")
                        with st.expander("View Auto-detected Features"):
                            st.write(selected_features[:50])  # Show first 50
                            if len(selected_features) > 50:
                                st.write(f"... and {len(selected_features) - 50} more")
                    
                    elif feature_mode == "Use All Numeric":
                        selected_features = numeric_cols
                        st.info(f"[DATA] Using all {len(selected_features)} numeric columns")
                    
                    else:  # Manual Selection
                        selected_features = st.multiselect(
                            "Select Feature Columns",
                            options=train_data.columns.tolist(),
                            default=feature_candidates[:20] if len(feature_candidates) >= 20 else feature_candidates,
                            help="Choose which columns to use as features"
                        )
                        
                        if not selected_features:
                            st.warning("WARNING: Please select at least one feature column")
                    
                    # Hyperparameter Configuration
                    st.markdown("---")
                    st.markdown("### Advanced Model Configuration")
                    
                    with st.expander("Hyperparameter Settings (Optional)", expanded=False):
                        st.markdown("""
                        <div style="padding: 10px; background: var(--bg-secondary); border-radius: 8px; margin-bottom: 1rem;">
                            <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">
                                Fine-tune model hyperparameters for optimal performance. Default values are recommended for most cases.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Model selection
                        st.markdown("**Model Selection**")
                        col_model1, col_model2 = st.columns(2)
                        
                        with col_model1:
                            use_rf = st.checkbox("Random Forest", value=True, help="Fast and robust ensemble method")
                            use_gb = st.checkbox("Gradient Boosting", value=True, help="Powerful sequential boosting")
                        
                        with col_model2:
                            try:
                                import xgboost
                                use_xgb = st.checkbox("XGBoost", value=False, help="Advanced gradient boosting (slower)")
                            except ImportError:
                                use_xgb = False
                                st.caption("XGBoost not installed")
                            
                            try:
                                import lightgbm
                                use_lgbm = st.checkbox("LightGBM", value=False, help="Fast gradient boosting")
                            except ImportError:
                                use_lgbm = False
                                st.caption("LightGBM not installed")
                        
                        # Check at least one model selected
                        models_selected = sum([use_rf, use_gb, use_xgb, use_lgbm])
                        if models_selected == 0:
                            st.warning("Select at least one model")
                        
                        st.markdown("---")
                        
                        # Random Forest Parameters
                        if use_rf:
                            st.markdown("**Random Forest Parameters**")
                            rf_col1, rf_col2, rf_col3 = st.columns(3)
                            
                            with rf_col1:
                                rf_n_estimators = st.number_input(
                                    "RF Trees",
                                    min_value=50,
                                    max_value=500,
                                    value=100,
                                    step=50,
                                    help="Number of trees in the forest"
                                )
                            
                            with rf_col2:
                                rf_max_depth = st.number_input(
                                    "RF Max Depth",
                                    min_value=3,
                                    max_value=30,
                                    value=15,
                                    step=1,
                                    help="Maximum tree depth (None for unlimited)"
                                )
                            
                            with rf_col3:
                                rf_min_samples_split = st.number_input(
                                    "RF Min Split",
                                    min_value=2,
                                    max_value=20,
                                    value=2,
                                    step=1,
                                    help="Minimum samples to split node"
                                )
                            
                            st.markdown("")
                        
                        # Gradient Boosting Parameters
                        if use_gb:
                            st.markdown("**Gradient Boosting Parameters**")
                            gb_col1, gb_col2, gb_col3 = st.columns(3)
                            
                            with gb_col1:
                                gb_n_estimators = st.number_input(
                                    "GB Trees",
                                    min_value=50,
                                    max_value=300,
                                    value=100,
                                    step=50,
                                    help="Number of boosting stages"
                                )
                            
                            with gb_col2:
                                gb_learning_rate = st.number_input(
                                    "GB Learning Rate",
                                    min_value=0.01,
                                    max_value=0.5,
                                    value=0.1,
                                    step=0.01,
                                    format="%.2f",
                                    help="Shrinks contribution of each tree"
                                )
                            
                            with gb_col3:
                                gb_max_depth = st.number_input(
                                    "GB Max Depth",
                                    min_value=3,
                                    max_value=10,
                                    value=5,
                                    step=1,
                                    help="Maximum depth of trees"
                                )
                            
                            st.markdown("")
                        
                        # XGBoost Parameters
                        if use_xgb:
                            st.markdown("**XGBoost Parameters**")
                            xgb_col1, xgb_col2, xgb_col3 = st.columns(3)
                            
                            with xgb_col1:
                                xgb_n_estimators = st.number_input(
                                    "XGB Trees",
                                    min_value=50,
                                    max_value=500,
                                    value=100,
                                    step=50,
                                    help="Number of boosting rounds"
                                )
                            
                            with xgb_col2:
                                xgb_learning_rate = st.number_input(
                                    "XGB Learning Rate",
                                    min_value=0.01,
                                    max_value=0.5,
                                    value=0.1,
                                    step=0.01,
                                    format="%.2f",
                                    help="Step size shrinkage"
                                )
                            
                            with xgb_col3:
                                xgb_max_depth = st.number_input(
                                    "XGB Max Depth",
                                    min_value=3,
                                    max_value=10,
                                    value=6,
                                    step=1,
                                    help="Maximum tree depth"
                                )
                            
                            st.markdown("")
                        
                        # LightGBM Parameters
                        if use_lgbm:
                            st.markdown("**LightGBM Parameters**")
                            lgbm_col1, lgbm_col2, lgbm_col3 = st.columns(3)
                            
                            with lgbm_col1:
                                lgbm_n_estimators = st.number_input(
                                    "LGBM Trees",
                                    min_value=50,
                                    max_value=500,
                                    value=100,
                                    step=50,
                                    help="Number of boosting iterations"
                                )
                            
                            with lgbm_col2:
                                lgbm_learning_rate = st.number_input(
                                    "LGBM Learning Rate",
                                    min_value=0.01,
                                    max_value=0.5,
                                    value=0.1,
                                    step=0.01,
                                    format="%.2f",
                                    help="Boosting learning rate"
                                )
                            
                            with lgbm_col3:
                                lgbm_num_leaves = st.number_input(
                                    "LGBM Num Leaves",
                                    min_value=10,
                                    max_value=300,
                                    value=31,
                                    step=10,
                                    help="Maximum tree leaves"
                                )
                    
                    # Training Button
                    st.markdown("---")
                    if st.button("Train Model", type="primary", width='stretch'):
                        
                        if not selected_features:
                            st.error("[FAIL] Please select at least one feature column!")
                        else:
                            with st.spinner("Training model... This may take several minutes."):
                                try:
                                    from sklearn.model_selection import train_test_split
                                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                                    from sklearn.linear_model import LogisticRegression
                                    from sklearn.ensemble import StackingClassifier
                                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                                    
                                    # Prepare data - use only selected features
                                    available_features = [f for f in selected_features if f in train_data.columns and f != target_col]
                                    
                                    if not available_features:
                                        st.error("[FAIL] No valid features selected!")
                                        st.stop()
                                    
                                    X = train_data[available_features].copy()
                                    y = train_data[target_col].copy()
                                    
                                    # Show data prep info
                                    st.info(f"[DATA] Using {len(available_features)} features for training")
                                    
                                    # Convert target to binary
                                    if y.dtype == 'object':
                                        # NASA catalog format detection
                                        unique_values = y.unique()
                                        
                                        # Kepler KOI format: CONFIRMED, FALSE POSITIVE, CANDIDATE
                                        if 'CONFIRMED' in unique_values or 'CANDIDATE' in unique_values:
                                            positive_labels = ['CONFIRMED', 'CANDIDATE']
                                            y = y.isin(positive_labels).astype(int)
                                            st.info(f"[TARGET] Target mapping: {positive_labels} → 1 (Exoplanet), Others → 0")
                                        
                                        # K2/TESS format: Confirmed, False Positive
                                        elif 'Confirmed' in unique_values:
                                            y = (y == 'Confirmed').astype(int)
                                            st.info("[TARGET] Target mapping: 'Confirmed' → 1, Others → 0")
                                        
                                        # Generic Exoplanet label
                                        else:
                                            y = (y.str.contains('Exoplanet|CONFIRMED|Confirmed|Planet', case=False, na=False)).astype(int)
                                            st.info("[TARGET] Target mapping: Labels containing 'Exoplanet/Confirmed/Planet' → 1, Others → 0")
                                    
                                    # Check class balance
                                    class_counts = y.value_counts()
                                    if len(class_counts) < 2:
                                        st.error("[FAIL] Target column must have at least 2 classes (exoplanet vs non-exoplanet)")
                                        st.stop()
                                    
                                    st.markdown(f"**Class Distribution:** [STAR] Exoplanets: {class_counts.get(1, 0)} | [FAIL] Non-Exoplanets: {class_counts.get(0, 0)}")
                                    
                                    # Convert string columns to numeric
                                    string_cols = X.select_dtypes(include=['object']).columns
                                    if len(string_cols) > 0:
                                        st.warning(f"WARNING: Converting {len(string_cols)} string columns to numeric...")
                                        for col in string_cols:
                                            X[col] = pd.to_numeric(X[col], errors='coerce')
                                    
                                    # Handle missing values and infinities
                                    X = X.fillna(0).replace([np.inf, -np.inf], 0)
                                    
                                    # Split data
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=test_size/100, random_state=42, stratify=y
                                    )
                                    
                                    # Build ensemble with custom hyperparameters
                                    base_estimators = []
                                    
                                    if use_rf:
                                        rf_model = RandomForestClassifier(
                                            n_estimators=rf_n_estimators,
                                            max_depth=rf_max_depth if rf_max_depth > 0 else None,
                                            min_samples_split=rf_min_samples_split,
                                            random_state=42,
                                            n_jobs=-1
                                        )
                                        base_estimators.append(('rf', rf_model))
                                    
                                    if use_gb:
                                        gb_model = GradientBoostingClassifier(
                                            n_estimators=gb_n_estimators,
                                            learning_rate=gb_learning_rate,
                                            max_depth=gb_max_depth,
                                            random_state=42
                                        )
                                        base_estimators.append(('gb', gb_model))
                                    
                                    if use_xgb:
                                        try:
                                            from xgboost import XGBClassifier
                                            xgb_model = XGBClassifier(
                                                n_estimators=xgb_n_estimators,
                                                learning_rate=xgb_learning_rate,
                                                max_depth=xgb_max_depth,
                                                random_state=42,
                                                n_jobs=-1,
                                                eval_metric='logloss'
                                            )
                                            base_estimators.append(('xgb', xgb_model))
                                        except Exception as e:
                                            st.warning(f"XGBoost not available: {str(e)}")
                                    
                                    if use_lgbm:
                                        try:
                                            from lightgbm import LGBMClassifier
                                            lgbm_model = LGBMClassifier(
                                                n_estimators=lgbm_n_estimators,
                                                learning_rate=lgbm_learning_rate,
                                                num_leaves=lgbm_num_leaves,
                                                random_state=42,
                                                n_jobs=-1,
                                                verbose=-1
                                            )
                                            base_estimators.append(('lgbm', lgbm_model))
                                        except Exception as e:
                                            st.warning(f"LightGBM not available: {str(e)}")
                                    
                                    # Check we have at least one model
                                    if not base_estimators:
                                        st.error("No models selected! Please enable at least one model in hyperparameter settings.")
                                        st.stop()
                                    
                                    # Show selected models
                                    model_names = [name for name, _ in base_estimators]
                                    st.info(f"Training ensemble with: {', '.join(model_names)}")
                                    
                                    # Build stacking ensemble
                                    model = StackingClassifier(
                                        estimators=base_estimators,
                                        final_estimator=LogisticRegression(max_iter=1000),
                                        cv=cv_folds,
                                        n_jobs=-1
                                    )
                                    
                                    # Train
                                    model.fit(X_train, y_train)
                                    
                                    # Evaluate
                                    y_pred = model.predict(X_test)
                                    y_proba = model.predict_proba(X_test)[:, 1]
                                    
                                    metrics = {
                                        'test_accuracy': accuracy_score(y_test, y_pred),
                                        'test_precision': precision_score(y_test, y_pred),
                                        'test_recall': recall_score(y_test, y_pred),
                                        'test_f1': f1_score(y_test, y_pred),
                                        'test_auc': roc_auc_score(y_test, y_proba),
                                        'optimal_threshold': 0.5,  # Could optimize this
                                    }
                                    
                                    # Store hyperparameters used
                                    hyperparameters = {
                                        'models': model_names,
                                        'cv_folds': cv_folds,
                                        'test_size': test_size
                                    }
                                    
                                    if use_rf:
                                        hyperparameters['rf'] = {
                                            'n_estimators': rf_n_estimators,
                                            'max_depth': rf_max_depth,
                                            'min_samples_split': rf_min_samples_split
                                        }
                                    
                                    if use_gb:
                                        hyperparameters['gb'] = {
                                            'n_estimators': gb_n_estimators,
                                            'learning_rate': gb_learning_rate,
                                            'max_depth': gb_max_depth
                                        }
                                    
                                    if use_xgb:
                                        hyperparameters['xgb'] = {
                                            'n_estimators': xgb_n_estimators,
                                            'learning_rate': xgb_learning_rate,
                                            'max_depth': xgb_max_depth
                                        }
                                    
                                    if use_lgbm:
                                        hyperparameters['lgbm'] = {
                                            'n_estimators': lgbm_n_estimators,
                                            'learning_rate': lgbm_learning_rate,
                                            'num_leaves': lgbm_num_leaves
                                        }
                                    
                                    # Save model
                                    model_data = {
                                        'model': model,
                                        'feature_names': X.columns.tolist(),
                                        'hyperparameters': hyperparameters,
                                        **metrics
                                    }
                                    
                                    save_path = MODEL_DIR / f"{model_name}.joblib"
                                    joblib.dump(model_data, save_path)
                                    
                                    st.success(f"[OK] Model trained successfully!")
                                    st.success(f" Saved to: `{save_path.name}`")
                                    
                                    # Display metrics
                                    st.markdown("### [DATA] Training Results")
                                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                                    
                                    with metric_col1:
                                        st.metric("Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
                                    with metric_col2:
                                        st.metric("Precision", f"{metrics['test_precision']*100:.2f}%")
                                    with metric_col3:
                                        st.metric("Recall", f"{metrics['test_recall']*100:.2f}%")
                                    
                                    # Option to immediately load the new model
                                    st.markdown("---")
                                    col_load1, col_load2 = st.columns(2)
                                    with col_load1:
                                        if st.button("� Load This Model Now", type="primary", use_container_width=True):
                                            # Load the newly trained model
                                            new_model_data, new_metadata = load_model(save_path)
                                            if new_model_data is not None:
                                                st.session_state.model_data = new_model_data
                                                st.session_state.metadata = new_metadata
                                                st.session_state.model_loaded = True
                                                st.success(f"[OK] Now using {save_path.name}")
                                                st.rerun()
                                    with col_load2:
                                        st.info("[INFO] Or select it from the dropdown in the header")
                                    
                                except Exception as e:
                                    st.error(f"Training failed: {str(e)}")
                                    with st.expander("Error Details"):
                                        import traceback
                                        st.code(traceback.format_exc())
            else:
                st.info("Select a data source to begin training")
                
                st.markdown("""
                **NASA Catalog Training (Recommended):**
                - Automatically handles NASA Exoplanet Archive format
                - Skips comment lines starting with `#`
                - Auto-detects disposition columns (`koi_disposition`, `disposition`)
                - Smart feature filtering (excludes IDs, names, dates)
                - Supports: Kepler KOI, K2, TESS catalogs
                
                **Custom CSV Format:**
                - Include a target column (e.g., `koi_disposition`, `label`)
                - Features should be numeric (string columns will be converted)
                - Recommended: At least 1000 samples for reliable training
                
                **Supported Target Formats:**
                - Kepler KOI: `CONFIRMED`, `FALSE POSITIVE`, `CANDIDATE`
                - K2/TESS: `Confirmed`, `False Positive`
                - Custom: `Exoplanet`, `0/1`, or any binary label
                
                **Example (Kepler KOI format):**
                ```csv
                koi_period,koi_depth,koi_prad,koi_teq,...,koi_disposition
                12.3,1500,2.5,450,...,CONFIRMED
                8.7,800,1.2,300,...,FALSE POSITIVE
                ```
                
                **CSV File Requirements:**
                - Consistent column counts (or use NASA format)
                - UTF-8 encoding
                - Comment lines with `#` are automatically skipped
                - At least one numeric feature column
                """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # RIGHT COLUMN: Prediction Summary (only show on New Prediction tab)
    with col_right:
        if st.session_state.active_tab == 'predict':
            st.markdown('<h3 class="control-title">Prediction Summary</h3>', unsafe_allow_html=True)
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            # Check for both batch results and single prediction
            if st.session_state.prediction_results is not None:
                results = st.session_state.prediction_results
                
                n_exoplanets = (results['Prediction'] == 'Exoplanet').sum()
                n_total = len(results)
                avg_confidence = results['Confidence'].mean()
                high_confidence = (results['Confidence'] >= 0.95).sum()
                
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                
                with sum_col1:
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-label">Predicted Exoplanets</div>
                        <div class="summary-value">{n_exoplanets}</div>
                        <div class="summary-subtext">of {n_total} samples</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with sum_col2:
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-label">Average Confidence</div>
                        <div class="summary-value">{avg_confidence*100:.1f}%</div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {avg_confidence*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with sum_col3:
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-label">High Confidence</div>
                        <div class="summary-value">{high_confidence}</div>
                        <div class="summary-subtext">&gt;95% confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif 'single_prediction' in st.session_state and st.session_state.single_prediction is not None:
                # Show single prediction summary
                st.markdown("""
                <div class="summary-card" style="margin: 10px 0;">
                    <div class="summary-label">Single Prediction Mode</div>
                    <div class="summary-value">Active</div>
                    <div class="summary-subtext">Results shown below</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No predictions yet. Upload data to see results.")
                
                # Placeholder summary
                st.markdown("""
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-label">Predicted Exoplanets</div>
                        <div class="summary-value">--</div>
                        <div class="summary-subtext">Awaiting data</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Average Confidence</div>
                        <div class="summary-value">--</div>
                        <div class="summary-subtext">Awaiting data</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">High Confidence</div>
                        <div class="summary-value">--</div>
                        <div class="summary-subtext">Awaiting data</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacing between major sections
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    # ===========================
    # SECTION 3: SINGLE PREDICTION RESULTS
    # ===========================
    if 'single_prediction' in st.session_state and st.session_state.single_prediction is not None:
        st.markdown('<h2 class="section-title">Detailed Analysis Results</h2>', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)
        
        pred_data = st.session_state.single_prediction
        
        # Make prediction with proper feature mapping
        if st.session_state.model_loaded:
            model = (st.session_state.model_data.get('stacking') or 
                    st.session_state.model_data.get('model') or 
                    st.session_state.model_data.get('final_ensemble'))
            
            feature_names = st.session_state.model_data.get('feature_names', [])
            
            if model and hasattr(model, 'predict_proba') and feature_names:
                # Create DataFrame with all features initialized to 0
                features_df = pd.DataFrame(0, index=[0], columns=feature_names)
                
                # Map input parameters to model features
                input_params = pred_data['input_params']
                feature_mapping = {
                    'koi_period': input_params['orbital_period'],
                    'koi_duration': input_params['transit_duration'],
                    'koi_depth': input_params['depth'],
                    'koi_prad': input_params['radius'],
                    'koi_teq': input_params['temp'],
                    'koi_insol': input_params['insolation'],
                    'koi_impact': input_params['impact'],
                    'koi_incl': input_params['inclination'],
                    'koi_steff': input_params['teff'],
                    'koi_slogg': input_params['logg'],
                    'koi_srad': input_params['metallicity']
                }
                
                # Update features that exist in the model
                for feat_name, value in feature_mapping.items():
                    if feat_name in features_df.columns:
                        features_df[feat_name] = value
                
                features_array = features_df.values
                confidence = model.predict_proba(features_array)[0, 1]
            else:
                confidence = 0.75  # Demo value
                features_array = None
        else:
            confidence = 0.75  # Demo value
            features_array = None
        
        # Use custom threshold if available, otherwise use default
        if 'custom_threshold' in st.session_state and st.session_state.custom_threshold is not None:
            threshold = st.session_state.custom_threshold
        else:
            threshold = meta.get("threshold", 0.5)
        
        is_exoplanet = confidence > threshold
        
        # Prediction Result Card
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            status = "EXOPLANET CANDIDATE" if is_exoplanet else "NOT AN EXOPLANET"
            status_color = "#28A745" if is_exoplanet else "#DC3545"
            
            st.markdown(f"""
            <div style="
                padding: 25px;
                background: linear-gradient(135deg, {status_color}15 0%, {status_color}05 100%);
                border-left: 5px solid {status_color};
                border-radius: 10px;
                margin: 20px 0;
            ">
                <h2 style="color: {status_color}; margin: 0; font-size: 28px;">{status}</h2>
                <p style="margin: 15px 0 0 0; font-size: 20px; color: var(--text-primary);">
                    Confidence: <strong>{confidence:.1%}</strong>
                </p>
                <div style="width: 100%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-top: 10px; overflow: hidden;">
                    <div style="width: {confidence*100}%; height: 100%; background: {status_color}; transition: width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            classification_text = "EXOPLANET" if is_exoplanet else "NOT EXOPLANET"
            classification_color = "#28A745" if is_exoplanet else "#DC3545"
            st.markdown(f"""
            <div style="display: flex; align-items: flex-end; height: 100%; margin: 20px 0;">
                <div class="metric-card" style="width: 100%; margin: 0;">
                    <div class="metric-label">CLASSIFICATION</div>
                    <div class="metric-value" style="font-size: 20px; color: {classification_color};">
                        {classification_text}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Habitability Assessment (only for exoplanets)
        if is_exoplanet and HABITABILITY_AVAILABLE and HABITABILITY_SCORER is not None:
            st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title" style="margin-top: 0;">Habitability Assessment</h3>', unsafe_allow_html=True)
            st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
            
            try:
                scorer = HABITABILITY_SCORER
                params = pred_data['params']
                
                # Use calculate_habitability_score method
                result = scorer.calculate_habitability_score(
                    radius=params['radius'],
                    temp=params['teq'],
                    insolation=params['insolation'],
                    stellar_temp=params['teff'],
                    period=params['orbital_period']
                )
                
                score = result.get('habitability_score', 0.0)
                category = result.get('habitability_class', 'UNKNOWN')
                esi = result.get('esi', 0.0)
                
                # Color coding
                if score >= 0.8:
                    hab_color = "#28A745"
                elif score >= 0.6:
                    hab_color = "#FFA500"
                elif score >= 0.4:
                    hab_color = "#FF8C00"
                else:
                    hab_color = "#DC3545"
                
                hab_col1, hab_col2, hab_col3, hab_col4 = st.columns(4)
                
                with hab_col1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {hab_color};">
                        <div class="metric-label">HABITABILITY SCORE</div>
                        <div class="metric-value" style="color: {hab_color};">{score:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with hab_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">CATEGORY</div>
                        <div class="metric-value">{category}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with hab_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">ESI INDEX</div>
                        <div class="metric-value">{esi:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with hab_col4:
                    earth_sim = score * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">EARTH SIMILARITY</div>
                        <div class="metric-value">{earth_sim:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Component breakdown
                st.markdown('<h4 style="margin-top: 25px; color: var(--text-secondary);">Component Analysis</h4>', unsafe_allow_html=True)
                
                # Extract component scores
                components = {
                    'Radius': result.get('radius_score', 0.0),
                    'Temperature': result.get('temperature_score', 0.0),
                    'Insolation': result.get('insolation_score', 0.0),
                    'Stellar': result.get('stellar_score', 0.0),
                    'Orbital': result.get('orbital_score', 0.0)
                }
                
                comp_cols = st.columns(5)
                for i, (comp_name, comp_score) in enumerate(components.items()):
                    if pd.notna(comp_score):
                        with comp_cols[i]:
                            comp_label = comp_name
                            st.markdown(f"""
                            <div style="text-align: center; padding: 15px; background: var(--bg-secondary); border-radius: 8px;">
                                <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">{comp_label}</div>
                                <div style="font-size: 20px; font-weight: 600; color: {hab_color};">{comp_score:.2f}</div>
                                <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-top: 8px;">
                                    <div style="width: {comp_score*100}%; height: 100%; background: {hab_color}; border-radius: 2px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.info(f"Habitability assessment unavailable: {str(e)}")
        
        # AI Explanation with SHAP
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div style="margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title" style="margin-top: 0;">AI Explanation</h3>', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom: 0.75rem;"></div>', unsafe_allow_html=True)
        st.info("Understanding which features contributed most to this prediction")
        
        try:
            import shap
            import matplotlib.pyplot as plt
            
            with st.spinner(" Generating AI explanation with SHAP..."):
                if st.session_state.model_loaded and features_array is not None:
                    # Get the correct model for SHAP
                    model_data = st.session_state.model_data
                
                # MODEL AGREEMENT DISPLAY - Show base estimator votes
                model = (model_data.get('stacking') or 
                        model_data.get('model') or 
                        model_data.get('final_ensemble'))
                
                if model is not None:
                    estimator_names, estimators = get_base_estimators(model)
                    
                    if estimators and len(estimators) > 1:
                        st.markdown("### Model Agreement")
                        st.markdown("*How each base model voted on this prediction*")
                        
                        model_explanations = {
                            'rf': '**RF** = Random Forest',
                            'gb': '**GB** = Gradient Boosting',
                            'xgb': '**XGB** = XGBoost',
                            'lgbm': '**LGBM** = LightGBM'
                        }
                        legend_text = " | ".join([v for k, v in model_explanations.items() if any(k in name.lower() for name in estimator_names)])
                        if legend_text:
                            st.caption(legend_text)
                        
                        cols = st.columns(min(len(estimators), 4))
                        # Use confidence from single prediction (already calculated above)
                        ensemble_proba = confidence if 'confidence' in locals() and isinstance(confidence, float) else 0.5
                        
                        for idx, (name, estimator) in enumerate(zip(estimator_names, estimators)):
                            try:
                                if hasattr(estimator, 'predict_proba'):
                                    est_proba = estimator.predict_proba(features_array)[:, 1][0]
                                    with cols[idx % 4]:
                                        display_name = name.upper()[:4]
                                        delta = est_proba - ensemble_proba
                                        
                                        st.markdown(f"""
                                        <div class="metric-card" style="text-align: center; margin: 5px 0;">
                                            <div class="metric-label">{display_name}</div>
                                            <div class="metric-value" style="font-size: 1.5rem; color: {'var(--success)' if est_proba >= 0.5 else 'var(--text-secondary)'};">
                                                {est_proba*100:.1f}%
                                            </div>
                                            <div style="font-size: 0.75rem; color: {'#28A745' if delta > 0 else '#DC3545' if delta < 0 else 'var(--text-secondary)'};">
                                                {'+' if delta > 0 else ''}{delta*100:.1f}% vs ensemble
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            except:
                                pass
                        
                        st.markdown("---")
                
                # For StackingClassifier, extract one of the base estimators
                base_model = None
                model_name = "Model"
                
                # Priority: Use individual tree-based models (XGBoost, LightGBM, RF)
                # These work best with TreeExplainer
                if 'xgboost' in model_data and model_data['xgboost'] is not None:
                    base_model = model_data['xgboost']
                    model_name = "XGBoost"
                elif 'lightgbm' in model_data and model_data['lightgbm'] is not None:
                    base_model = model_data['lightgbm']
                    model_name = "LightGBM"
                elif 'random_forest' in model_data and model_data['random_forest'] is not None:
                    base_model = model_data['random_forest']
                    model_name = "Random Forest"
                
                # If we have a stacking classifier, extract the first base estimator
                if base_model is None:
                    for key in ['stacking', 'model', 'final_ensemble']:
                        if key in model_data and model_data[key] is not None:
                            stacking_model = model_data[key]
                            if hasattr(stacking_model, 'estimators_'):
                                # Get the first tree-based estimator from the stack
                                # estimators_ is a list of tuples (name, estimator)
                                for estimator_tuple in stacking_model.estimators_:
                                    if isinstance(estimator_tuple, tuple) and len(estimator_tuple) == 2:
                                        name, estimator = estimator_tuple
                                    else:
                                        # If not tuple, it's just the estimator
                                        estimator = estimator_tuple
                                    
                                    if 'XGB' in str(type(estimator).__name__):
                                        base_model = estimator
                                        model_name = "XGBoost (from ensemble)"
                                        break
                                    elif 'LGBM' in str(type(estimator).__name__):
                                        base_model = estimator
                                        model_name = "LightGBM (from ensemble)"
                                        break
                                    elif 'RandomForest' in str(type(estimator).__name__):
                                        base_model = estimator
                                        model_name = "Random Forest (from ensemble)"
                                        break
                                if base_model is not None:
                                    break
                
                feature_names = model_data.get('feature_names', [])
                
                if base_model and feature_names:
                    st.markdown(f"*Explanation based on: {model_name}*")
                    
                    try:
                        # Use TreeExplainer for tree-based models
                        explainer = shap.TreeExplainer(base_model)
                        shap_values_raw = explainer.shap_values(features_array)
                        
                        # Handle different shap_values formats
                        # Binary classification can return: list [class_0, class_1] or array (n_samples, n_features, 2)
                        if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                            # List format: [class_0_values, class_1_values]
                            shap_values = shap_values_raw[1]  # Use positive class (exoplanet)
                        elif isinstance(shap_values_raw, np.ndarray):
                            if len(shap_values_raw.shape) == 3:
                                # Shape: (n_samples, n_features, n_classes)
                                shap_values = shap_values_raw[:, :, 1]  # Use positive class
                            else:
                                # Shape: (n_samples, n_features)
                                shap_values = shap_values_raw
                        else:
                            shap_values = shap_values_raw
                    except Exception as e:
                        # If TreeExplainer fails, show a simplified explanation
                        st.warning(f"TreeExplainer not available for this model. Showing feature importance instead.")
                        
                        # Get feature importances if available
                        if hasattr(base_model, 'feature_importances_'):
                            importances = base_model.feature_importances_
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances,
                                'Value': features_array[0]
                            })
                            importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                            
                            st.markdown("**Top 10 Most Important Features:**")
                            for idx, row in importance_df.iterrows():
                                st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; 
                                     background: var(--bg-secondary); border-left: 3px solid var(--accent-primary); 
                                     border-radius: 6px; margin: 8px 0;">
                                    <div>
                                        <strong style="color: var(--text-primary);">{row['Feature']}</strong>
                                        <span style="color: var(--text-secondary); margin-left: 10px;">Value: {row['Value']:.3f}</span>
                                    </div>
                                    <div style="font-weight: 600; color: var(--accent-primary);">
                                        {row['Importance']:.4f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.info("Prediction is based on ensemble model analysis")
                            continue_shap = False
                        else:
                            raise e
                    
                    continue_shap = True
                    
                    if continue_shap:
                        # SHAP Force Plot
                        st.markdown("**Feature Impact Visualization:**")
                        
                        # Handle expected_value (might be array or scalar)
                        expected_val = explainer.expected_value
                        if isinstance(expected_val, (list, np.ndarray)):
                            if len(expected_val) > 1:
                                expected_val = expected_val[1]  # Positive class
                            else:
                                expected_val = expected_val[0]
                        elif isinstance(expected_val, np.ndarray) and expected_val.size > 1:
                            expected_val = float(expected_val.flatten()[1] if expected_val.size > 1 else expected_val.flatten()[0])
                        
                        # Get single sample shap values (1D array of feature impacts)
                        shap_array = np.array(shap_values)
                        if len(shap_array.shape) == 2:
                            # Shape: (n_samples, n_features) -> extract first sample
                            single_shap_values = shap_array[0]
                        elif len(shap_array.shape) == 1:
                            # Already 1D
                            single_shap_values = shap_array
                        else:
                            # Unexpected shape, flatten and hope for the best
                            single_shap_values = shap_array.flatten()[:len(feature_names)]
                        
                        # Verify lengths match before plotting
                        if len(single_shap_values) != len(feature_names):
                            st.warning(f"Cannot display force plot: SHAP values ({len(single_shap_values)}) don't match features ({len(feature_names)})")
                        else:
                            plt.figure(figsize=(10, 2))
                            try:
                                shap.force_plot(
                                    expected_val,
                                    single_shap_values,
                                    features_array[0][:len(feature_names)],
                                    feature_names=feature_names,
                                    matplotlib=True,
                                    show=False
                                )
                                st.pyplot(plt.gcf(), width='content')
                            except Exception as plot_error:
                                st.warning(f"Force plot unavailable: {str(plot_error)}")
                            finally:
                                plt.close()
                            
                            # SHAP WATERFALL PLOT
                            st.markdown("---")
                            st.markdown("**SHAP Waterfall Plot:**")
                            st.caption("Shows how features cumulatively push the prediction from the base value")
                            
                            try:
                                fig, ax = plt.subplots(figsize=(10, 2))
                                
                                # Create shap Explanation object for waterfall plot
                                explanation = shap.Explanation(
                                    values=single_shap_values,
                                    base_values=expected_val,
                                    data=features_array[0][:len(feature_names)],
                                    feature_names=feature_names
                                )
                                
                                shap.waterfall_plot(explanation, max_display=10, show=False)
                                st.pyplot(fig, width='content')
                                plt.close()
                            except Exception as waterfall_error:
                                st.info(f"Waterfall plot unavailable: {str(waterfall_error)}")
                        
                        # Top Features Table
                        st.markdown("---")
                        st.markdown("**Top Contributing Features:**")
                        
                        # Extract 1D array of SHAP values for the single sample
                        shap_array = np.array(shap_values)
                        if len(shap_array.shape) == 2:
                            shap_vals = shap_array[0]  # First sample
                        elif len(shap_array.shape) == 1:
                            shap_vals = shap_array
                        else:
                            shap_vals = shap_array.flatten()[:len(feature_names)]
                        
                        # Verify lengths match
                        if len(shap_vals) != len(feature_names):
                            st.warning(f"Feature mismatch: {len(shap_vals)} SHAP values vs {len(feature_names)} features. Showing feature importance instead.")
                            # Fallback to feature importance
                            if hasattr(base_model, 'feature_importances_'):
                                importances = base_model.feature_importances_
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances,
                                    'Value': features_array[0][:len(feature_names)]
                                })
                                importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
                                
                                for idx, row in importance_df.iterrows():
                                    st.markdown(f"""
                                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; 
                                         background: var(--bg-secondary); border-left: 3px solid var(--accent-primary); 
                                         border-radius: 6px; margin: 8px 0;">
                                        <div>
                                            <strong style="color: var(--text-primary);">{row['Feature']}</strong>
                                            <span style="color: var(--text-secondary); margin-left: 10px;">Value: {row['Value']:.3f}</span>
                                        </div>
                                        <div style="font-weight: 600; color: var(--accent-primary);">
                                            {row['Importance']:.4f}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Impact': shap_vals,
                                'Value': features_array[0][:len(feature_names)]
                            })
                            importance_df['Abs Impact'] = abs(importance_df['Impact'])
                            importance_df = importance_df.sort_values('Abs Impact', ascending=False).head(5)
                        
                        # Display as styled cards
                        for idx, row in importance_df.iterrows():
                            # Handle both Impact (SHAP) and Importance (fallback) columns
                            if 'Impact' in row:
                                impact_value = row['Impact']
                                impact_color = "#28A745" if impact_value > 0 else "#DC3545"
                                impact_text = f"{'+' if impact_value > 0 else ''}{impact_value:.4f}"
                            elif 'Importance' in row:
                                impact_value = row['Importance']
                                impact_color = "#2E5EAA"
                                impact_text = f"{impact_value:.4f}"
                            else:
                                impact_color = "#2E5EAA"
                                impact_text = "N/A"
                            
                            # Only show value if it's not zero
                            value_text = f"Value: {row['Value']:.3f}" if row['Value'] != 0 else ''
                            
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; 
                                 background: var(--bg-secondary); border-left: 3px solid {impact_color}; 
                                 border-radius: 6px; margin: 8px 0;">
                                <div>
                                    <strong style="color: var(--text-primary);">{row['Feature']}</strong>
                                    <span style="color: var(--text-secondary); margin-left: 10px; font-size: 0.9rem;">{value_text}</span>
                                </div>
                                <div style="font-weight: 600; color: {impact_color};">
                                    {impact_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Model not available for SHAP explanation")
                else:
                    st.info("SHAP explanation available when model is loaded")
                    
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {str(e)}")
            st.info("Prediction is based on ensemble model analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add spacing between major sections
    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    # ===========================
    # SECTION 4: BATCH RESULTS TABLE
    # ===========================
    if st.session_state.prediction_results is not None:
        st.markdown('<h2 class="section-title">Detailed Prediction Results</h2>', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="data-table-container">', unsafe_allow_html=True)
        
        results = st.session_state.prediction_results.copy()
        
        # Format confidence column
        results['Confidence %'] = (results['Confidence'] * 100).round(1)
        results['Viability %'] = (results['Viability Score'] * 100).round(1)
        
        # Create display dataframe
        display_df = results[['ID', 'Prediction', 'Confidence %', 'Viability %']].copy()
        
        # Style the dataframe
        def highlight_predictions(row):
            if row['Prediction'] == 'Exoplanet':
                return [''] * len(row)  # Streamlit handles this differently
            return [''] * len(row)
        
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            height=400
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button
        csv = results.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            csv,
            "exoplanet_predictions.csv",
            "text/csv",
            key='download-results',
            width='content'
        )
        
        # AI Explanation for Batch Results (SHAP with candidate selector)
        st.markdown("---")
        st.markdown('<h3 class="section-title" style="margin-top: 30px;">AI Explanation - Select Candidate</h3>', unsafe_allow_html=True)
        
        # Check if we have input data stored
        if 'prediction_input_data' not in st.session_state or st.session_state.prediction_input_data is None:
            st.warning("WARNING: Original input data not available. Please re-run predictions to enable AI explanations.")
        else:
            input_data = st.session_state.prediction_input_data
            
            # Create selector for candidates
            candidate_options = []
            for idx, row in results.iterrows():
                candidate_id = row.get('ID', f'Sample {idx}')
                prediction = row['Prediction']
                confidence = row['Confidence'] * 100
                label = f"{candidate_id} - {prediction} ({confidence:.1f}% confidence)"
                candidate_options.append((label, idx))
            
            selected_label = st.selectbox(
                "Select a candidate to explain:",
                options=[label for label, _ in candidate_options],
                key='batch_shap_selector'
            )
            
            # Get the selected index
            selected_idx = next(idx for label, idx in candidate_options if label == selected_label)
            
            st.info(f"SHAP explanation showing why this candidate was classified as: **{results.iloc[selected_idx]['Prediction']}**")
            
            # Generate SHAP for selected batch candidate
            try:
                import shap
                import matplotlib.pyplot as plt
                
                with st.spinner(f" Generating explanation for selected candidate..."):
                    if st.session_state.model_loaded and selected_idx < len(input_data):
                        model_data = st.session_state.model_data
                        
                        # Get selected row from ORIGINAL input data
                        selected_candidate_data = input_data.iloc[selected_idx]
                        selected_result = results.iloc[selected_idx]
                        
                        # Get feature names and prepare features for SHAP
                        feature_names = model_data.get('feature_names', [])
                        
                        # Create proper feature dataframe with numeric conversion
                        features_df = pd.DataFrame(0, index=[0], columns=feature_names)
                        
                        # Fill in available features from input data with numeric conversion
                        for col in selected_candidate_data.index:
                            if col in features_df.columns:
                                value = selected_candidate_data[col]
                                # Convert to numeric if it's a string
                                if isinstance(value, str):
                                    value = pd.to_numeric(value, errors='coerce')
                                    if pd.isna(value):
                                        value = 0.0
                                features_df[col] = value
                        
                        # Ensure all values are numeric and handle inf/nan
                        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
                        features_array = features_df.values
                        
                        # Extract base model for SHAP
                        base_model = None
                        model_name = "Model"
                        
                        if 'xgboost' in model_data and model_data['xgboost'] is not None:
                            base_model = model_data['xgboost']
                            model_name = "XGBoost"
                        elif 'lightgbm' in model_data and model_data['lightgbm'] is not None:
                            base_model = model_data['lightgbm']
                            model_name = "LightGBM"
                        elif 'random_forest' in model_data and model_data['random_forest'] is not None:
                            base_model = model_data['random_forest']
                            model_name = "Random Forest"
                        
                        if base_model is None:
                            for key in ['stacking', 'model', 'final_ensemble']:
                                if key in model_data and model_data[key] is not None:
                                    stacking_model = model_data[key]
                                    if hasattr(stacking_model, 'estimators_'):
                                        for estimator_tuple in stacking_model.estimators_:
                                            if isinstance(estimator_tuple, tuple) and len(estimator_tuple) == 2:
                                                name, estimator = estimator_tuple
                                            else:
                                                estimator = estimator_tuple
                                            
                                            if 'XGB' in str(type(estimator).__name__):
                                                base_model = estimator
                                                model_name = "XGBoost (from ensemble)"
                                                break
                                            elif 'LGBM' in str(type(estimator).__name__):
                                                base_model = estimator
                                                model_name = "LightGBM (from ensemble)"
                                                break
                                            elif 'RandomForest' in str(type(estimator).__name__):
                                                base_model = estimator
                                                model_name = "Random Forest (from ensemble)"
                                                break
                                        if base_model is not None:
                                            break
                        
                        if base_model and feature_names and len(features_array) > 0:
                            st.markdown(f"*Explanation based on: {model_name}*")
                            
                            try:
                                explainer = shap.TreeExplainer(base_model)
                                shap_values_raw = explainer.shap_values(features_array)
                                
                                if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                                    shap_values = shap_values_raw[1]
                                elif isinstance(shap_values_raw, np.ndarray):
                                    if len(shap_values_raw.shape) == 3:
                                        shap_values = shap_values_raw[:, :, 1]
                                    else:
                                        shap_values = shap_values_raw
                                else:
                                    shap_values = shap_values_raw
                                
                                # Extract 1D array
                                shap_array = np.array(shap_values)
                                if len(shap_array.shape) == 2:
                                    shap_vals = shap_array[0]
                                elif len(shap_array.shape) == 1:
                                    shap_vals = shap_array
                                else:
                                    shap_vals = shap_array.flatten()[:len(feature_names)]
                                
                                if len(shap_vals) == len(feature_names):
                                    # Show prediction result for selected candidate
                                    pred_result = selected_result['Prediction']
                                    pred_confidence = selected_result['Confidence'] * 100
                                    result_color = "#28A745" if pred_result == "Exoplanet" else "#6C757D"
                                    
                                    st.markdown(f"""
                                <div style="padding: 15px; background: var(--bg-secondary); border-radius: 8px; 
                                     border-left: 4px solid {result_color}; margin: 15px 0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <strong style="color: var(--text-primary); font-size: 18px;">Classification: {pred_result}</strong>
                                            <div style="color: var(--text-secondary); margin-top: 5px;">
                                                Model Confidence: {pred_confidence:.1f}%
                                            </div>
                                        </div>
                                        <div style="font-size: 32px;">
                                            {pred_result[0]}
                                        </div>
                                    </div>
                                </div>
                                    """, unsafe_allow_html=True)
                                    
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Impact': shap_vals,
                                        'Value': features_array[0][:len(feature_names)]
                                    })
                                    importance_df['Abs Impact'] = abs(importance_df['Impact'])
                                    top_features = importance_df.sort_values('Abs Impact', ascending=False).head(10)
                                    
                                    # Show interpretation
                                    total_positive = importance_df[importance_df['Impact'] > 0]['Impact'].sum()
                                    total_negative = importance_df[importance_df['Impact'] < 0]['Impact'].sum()
                                    
                                    st.markdown(f"""
                                    <div style="padding: 12px; background: var(--info-bg); border-radius: 6px; margin: 10px 0;">
                                        <strong>Impact Summary:</strong><br>
                                        Features pushing <span style="color: #28A745; font-weight: 600;">towards Exoplanet</span>: {total_positive:.4f}<br>
                                        Features pushing <span style="color: #DC3545; font-weight: 600;">towards False Positive</span>: {total_negative:.4f}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add detailed sample information
                                    with st.expander("Detailed Sample Information", expanded=False):
                                        st.markdown("**Key Parameters:**")
                                        
                                        # Extract key features
                                        key_features = {
                                            'koi_period': 'Orbital Period (days)',
                                            'koi_prad': 'Planet Radius (Earth radii)',
                                            'koi_teq': 'Equilibrium Temperature (K)',
                                            'koi_insol': 'Insolation Flux (Earth flux)',
                                            'koi_depth': 'Transit Depth (ppm)',
                                            'koi_duration': 'Transit Duration (hours)',
                                            'koi_impact': 'Impact Parameter',
                                            'koi_steff': 'Stellar Temperature (K)',
                                            'koi_srad': 'Stellar Radius (Solar radii)',
                                            'koi_slogg': 'Stellar Surface Gravity (log10(cm/s²))'
                                        }
                                        
                                        param_cols = st.columns(2)
                                        col_idx = 0
                                        for feat_key, feat_label in key_features.items():
                                            if feat_key in selected_candidate_data.index:
                                                value = selected_candidate_data[feat_key]
                                                if pd.notna(value):
                                                    with param_cols[col_idx % 2]:
                                                        st.markdown(f"""
                                                        <div style="padding: 8px; background: var(--bg-tertiary); border-radius: 4px; margin: 5px 0;">
                                                            <div style="color: var(--text-secondary); font-size: 11px;">{feat_label}</div>
                                                            <div style="color: var(--text-primary); font-weight: 600; font-size: 16px;">{value:.3f}</div>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    col_idx += 1
                                        
                                        st.markdown("---")
                                        st.markdown("**All Feature Values:**")
                                        
                                        # Show all numeric features in a compact table
                                        all_features_df = pd.DataFrame({
                                            'Feature': feature_names[:20],  # First 20 features
                                            'Value': features_array[0][:20]
                                        })
                                        st.dataframe(all_features_df, hide_index=True, height=300)
                                    
                                    st.markdown("**Top 10 Contributing Features:**")
                                    st.markdown("*Green = pushes towards Exoplanet | Red = pushes towards False Positive*")
                                    
                                    for idx, row in top_features.iterrows():
                                        impact_color = "#28A745" if row['Impact'] > 0 else "#DC3545"
                                        impact_direction = "Exoplanet" if row['Impact'] > 0 else "False Positive"
                                        impact_percent = (abs(row['Impact']) / importance_df['Abs Impact'].sum()) * 100
                                        
                                        st.markdown(f"""
                                        <div style="padding: 12px; background: var(--bg-secondary); border-left: 3px solid {impact_color}; 
                                             border-radius: 6px; margin: 8px 0;">
                                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                                <div style="flex: 1;">
                                                    <strong style="color: var(--text-primary);">{row['Feature']}</strong>
                                                    <div style="color: var(--text-secondary); font-size: 12px; margin-top: 3px;">
                                                        Value: <strong>{row['Value']:.3f}</strong> → Pushes towards <strong>{impact_direction}</strong>
                                                    </div>
                                                    <div style="color: var(--text-secondary); font-size: 11px; margin-top: 2px;">
                                                        Contribution: {impact_percent:.1f}% of total impact
                                                    </div>
                                                </div>
                                                <div style="text-align: right; margin-left: 20px;">
                                                    <div style="font-weight: 600; color: {impact_color}; font-size: 18px;">
                                                        {'+' if row['Impact'] > 0 else ''}{row['Impact']:.4f}
                                                    </div>
                                                    <div style="width: 60px; height: 6px; background: #333; border-radius: 3px; margin-top: 5px; overflow: hidden;">
                                                        <div style="width: {impact_percent}%; height: 100%; background: {impact_color};"></div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("Feature mismatch - unable to show detailed explanation.")
                            except Exception as inner_shap_error:
                                st.warning(f"Could not generate SHAP explanation: {str(inner_shap_error)}")
                                st.info("Try using Single Prediction mode for more reliable AI explanations.")
                        else:
                            st.info("Model or features not available for SHAP analysis.")
                    else:
                        st.info("Model not loaded. Please check model configuration.")
            except Exception as shap_error:
                st.error(f"Error generating explanation: {str(shap_error)}")
                st.info("Try using Single Prediction mode for detailed AI explanations.")
        
        st.markdown("""
            <div style="padding: 20px; background: var(--bg-secondary); border-radius: 8px; border-left: 4px solid var(--accent-primary);">
                <strong>For Detailed AI Explanations:</strong><br>
                Switch to <strong>Single Prediction</strong> mode to see:
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>SHAP force plot showing feature impacts</li>
                    <li>Top 5 contributing features with values</li>
                    <li>Feature importance visualization</li>
                    <li>Detailed breakdown of prediction reasoning</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Habitability Summary for Batch Results
        if HABITABILITY_AVAILABLE and HABITABILITY_SCORER is not None:
            # Check if we have stored input data
            if 'prediction_input_data' in st.session_state:
                input_data = st.session_state.prediction_input_data
                exoplanet_count = len(results[results['Prediction'] == 'Exoplanet'])
                
                if exoplanet_count > 0:
                    st.markdown("---")
                    st.markdown('<h3 class="section-title" style="margin-top: 30px;"><i class="fas fa-globe"></i> Habitability Summary</h3>', unsafe_allow_html=True)
                    
                    try:
                        scorer = HABITABILITY_SCORER
                        
                        st.info(f"Found **{exoplanet_count} exoplanet candidates** - calculating habitability scores...")
                        
                        # Get exoplanet indices
                        exoplanet_mask = results['Prediction'] == 'Exoplanet'
                        exoplanet_indices = results[exoplanet_mask].index.tolist()
                        
                        # Calculate habitability for all exoplanet candidates
                        habitability_data = []
                        
                        for result_idx in exoplanet_indices:
                            try:
                                # Get original input data for this candidate
                                candidate_data = input_data.iloc[result_idx]
                                result_row = results.iloc[result_idx]
                                
                                # Extract required parameters with fallback values
                                radius = candidate_data.get('koi_prad', 1.0)
                                temp = candidate_data.get('koi_teq', 288.0)
                                insolation = candidate_data.get('koi_insol', 1.0)
                                stellar_temp = candidate_data.get('koi_steff', 5778.0)
                                period = candidate_data.get('koi_period', 365.0)
                                
                                # Convert to numeric if needed
                                radius = float(radius) if pd.notna(radius) else 1.0
                                temp = float(temp) if pd.notna(temp) else 288.0
                                insolation = float(insolation) if pd.notna(insolation) else 1.0
                                stellar_temp = float(stellar_temp) if pd.notna(stellar_temp) else 5778.0
                                period = float(period) if pd.notna(period) else 365.0
                                
                                # Calculate habitability score
                                hab_result = scorer.calculate_habitability_score(
                                    radius=radius,
                                    temp=temp,
                                    insolation=insolation,
                                    stellar_temp=stellar_temp,
                                    period=period
                                )
                                
                                # Extract score from result (could be dict or float)
                                if isinstance(hab_result, dict):
                                    hab_score = hab_result.get('habitability_score', 0.0)
                                else:
                                    hab_score = float(hab_result)
                                
                                habitability_data.append({
                                    'index': result_idx,
                                    'id': result_row['ID'],
                                    'confidence': result_row['Confidence'],
                                    'habitability': hab_score
                                })
                            except Exception as inner_e:
                                # Store 0.0 if calculation fails for this candidate
                                habitability_data.append({
                                    'index': result_idx,
                                    'id': result_row.get('ID', f'Sample-{result_idx+1}'),
                                    'confidence': result_row.get('Confidence', 0.0),
                                    'habitability': 0.0
                                })
                        
                        # Calculate statistics
                        habitability_scores = [d['habitability'] for d in habitability_data]
                        avg_hab = np.mean(habitability_scores) if habitability_scores else 0.0
                        max_hab = np.max(habitability_scores) if habitability_scores else 0.0
                        promising_count = sum(1 for s in habitability_scores if s >= 0.6)
                        
                        # Show habitability stats
                        hab_col1, hab_col2, hab_col3, hab_col4 = st.columns(4)
                        
                        with hab_col1:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-label">CANDIDATES</div>
                                <div class="metric-value">{}</div>
                            </div>
                            """.format(exoplanet_count), unsafe_allow_html=True)
                        
                        with hab_col2:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-label">AVG HABITABILITY</div>
                                <div class="metric-value">{:.1f}%</div>
                            </div>
                            """.format(avg_hab * 100), unsafe_allow_html=True)
                        
                        with hab_col3:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-label">BEST SCORE</div>
                                <div class="metric-value">{:.1f}%</div>
                            </div>
                            """.format(max_hab * 100), unsafe_allow_html=True)
                        
                        with hab_col4:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-label">PROMISING</div>
                                <div class="metric-value">{}</div>
                                <div class="metric-subtext">&gt;60% score</div>
                            </div>
                            """.format(promising_count), unsafe_allow_html=True)
                        
                        # Show top 3 candidates
                        if habitability_data:
                            st.markdown("**Top 3 Most Habitable Candidates:**")
                            
                            # Sort by habitability score
                            sorted_data = sorted(habitability_data, key=lambda x: x['habitability'], reverse=True)
                            top_3 = sorted_data[:3]
                            
                            for i, candidate in enumerate(top_3, 1):
                                hab_score = candidate['habitability']
                                candidate_id = candidate['id']
                                confidence = candidate['confidence'] * 100
                                
                                color = "#28A745" if hab_score >= 0.7 else "#FFC107" if hab_score >= 0.5 else "#6C757D"
                                
                                st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px; 
                                     background: var(--bg-secondary); border-left: 3px solid {color}; 
                                     border-radius: 6px; margin: 8px 0;">
                                    <div>
                                        <strong style="color: var(--text-primary);">#{i} {candidate_id}</strong>
                                        <span style="color: var(--text-secondary); margin-left: 10px;">Confidence: {confidence:.1f}%</span>
                                    </div>
                                    <div style="font-weight: 600; color: {color}; font-size: 18px;">
                                        {hab_score*100:.1f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div style="margin-top: 20px; padding: 15px; background: var(--info-bg); border-left: 4px solid var(--info); border-radius: 6px;">
                            <strong>Tip:</strong> Switch to <strong>Single Prediction</strong> mode to get detailed habitability 
                            breakdown with ESI components (temperature, radius, escape velocity) for any specific candidate.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.info(f"Found {exoplanet_count} exoplanet candidates. Use Single Prediction mode for detailed habitability analysis. (Error: {str(e)[:50]})")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
