"""
HEA (Habitable Exoplanet Analysis) - FastAPI Backend
NASA Space Apps Challenge 2025

A comprehensive REST API for exoplanet detection using ML models.
Features automatic Swagger documentation, SHAP explanations, and batch processing.

API Documentation available at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import io
import json
import shap
import warnings
warnings.filterwarnings('ignore')

# ===========================
# APP CONFIGURATION
# ===========================

app = FastAPI(
    title="HEA API - Habitable Exoplanet Analysis",
    description="""
     **NASA Space Apps Challenge 2025**
    
    High-performance Machine Learning API for exoplanet detection and analysis.
    
    ## Features
    
    * **Single & Batch Predictions**: Analyze one or multiple candidates
    * **Habitability Scoring**: Calculate habitability scores based on planetary characteristics
    * **SHAP Explanations**: Get AI explanations for predictions
    * **Model Insights**: Access model performance metrics and feature importance
    * **CSV Upload**: Upload CSV files for batch analysis
    * **Health Monitoring**: Check API and model status
    
    ## Model Performance
    
    * **Accuracy**: 94.29%
    * **Recall**: 98.00%
    * **Precision**: 94.03%
    * **AUC Score**: 0.97
    
    ## Quick Start
    
    1. Check health: `GET /health`
    2. Get model info: `GET /model/info`
    3. Make prediction: `POST /predict`
    4. Calculate habitability: `POST /habitability`
    5. Upload CSV: `POST /predict/batch/upload`
    """,
    version="1.0.0",
    contact={
        "name": "Felipe Cabrera",
        "email": "me@felieppe.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# GLOBAL VARIABLES
# ===========================

# Model directory - handle both running from backend/ and root directory
MODEL_DIR = Path(__file__).parent.parent / "models" / "catalog_models"
if not MODEL_DIR.exists():
    MODEL_DIR = Path("models/catalog_models")  # Fallback for root directory execution

model_cache = {
    "model": None,
    "metadata": None,
    "feature_names": [],
    "threshold": 0.5,
    "loaded_at": None
}

# ===========================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ===========================

class ExoplanetInput(BaseModel):
    """Single exoplanet candidate input"""
    data: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and their values",
        example={
            "koi_period": 3.5225,
            "koi_period_err1": 0.00012,
            "koi_time0bk": 134.452,
            "koi_depth": 200.5,
            "koi_duration": 2.5,
            "koi_count": 45,
            "koi_kepmag": 14.856,
            "st_teff": 5778,
            "st_rad": 1.02
        }
    )

class BatchExoplanetInput(BaseModel):
    """Batch prediction input"""
    data: List[Dict[str, float]] = Field(
        ...,
        description="List of candidates, each as a dictionary of features",
        example=[
            {"koi_period": 3.5225, "koi_depth": 200.5},
            {"koi_period": 4.8921, "koi_depth": 180.3}
        ]
    )

class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: str = Field(..., description="'Exoplanet' or 'Not Exoplanet'")
    probability: float = Field(..., description="Probability of being an exoplanet (0-1)")
    confidence: str = Field(..., description="Confidence level: 'Very High', 'High', 'Moderate', 'Low'")
    threshold: float = Field(..., description="Decision threshold used")
    timestamp: str = Field(..., description="Prediction timestamp")

class ExplanationResponse(BaseModel):
    """Prediction with SHAP explanation"""
    prediction: str
    probability: float
    confidence: str
    shap_explanation: Optional[Dict[str, Any]] = Field(
        None,
        description="SHAP values and feature importance"
    )
    base_model_votes: Optional[Dict[str, float]] = Field(
        None,
        description="Individual predictions from ensemble models"
    )
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics of the batch"
    )
    total_processed: int
    timestamp: str

class ModelInfo(BaseModel):
    """Model information and metrics"""
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    threshold: float
    n_features: int
    training_samples: int
    loaded_at: str
    version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    api_version: str
    timestamp: str

class HabitabilityInput(BaseModel):
    """Habitability scoring input"""
    radius: Optional[float] = Field(None, description="Planet radius in Earth radii", example=1.0)
    temp: Optional[float] = Field(None, description="Equilibrium temperature in Kelvin", example=288.0)
    insolation: Optional[float] = Field(None, description="Insolation flux (Earth = 1.0)", example=1.0)
    stellar_temp: Optional[float] = Field(None, description="Stellar temperature in Kelvin", example=5778.0)
    period: Optional[float] = Field(None, description="Orbital period in days", example=365.25)
    disposition: Optional[str] = Field(None, description="Candidate disposition", example="CONFIRMED")

class HabitabilityResponse(BaseModel):
    """Habitability scoring response"""
    habitability_score: float = Field(..., description="Overall habitability score (0-1)")
    habitability_class: str = Field(..., description="Classification: HIGH, MODERATE, LOW, VERY_LOW, UNKNOWN")
    radius_score: Optional[float] = Field(None, description="Radius similarity score")
    temperature_score: Optional[float] = Field(None, description="Temperature habitability score")
    insolation_score: Optional[float] = Field(None, description="Insolation flux score")
    stellar_score: Optional[float] = Field(None, description="Stellar properties score")
    orbital_score: Optional[float] = Field(None, description="Orbital characteristics score")
    esi: Optional[float] = Field(None, description="Earth Similarity Index")
    timestamp: str = Field(..., description="Calculation timestamp")

# ===========================
# UTILITY FUNCTIONS
# ===========================

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
        decimal_degrees = (hours + minutes/60 + seconds/3600) * 15
        return decimal_degrees
    except:
        return 0.0

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any string columns to numeric"""
    # Check for string columns
    string_cols = df.select_dtypes(include=['object']).columns
    
    for col in string_cols:
        # Special handling for RA string format
        if 'rastr' in col.lower() or col == 'rastr':
            df[col] = df[col].apply(convert_ra_to_decimal)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def load_model():
    """Load the trained model and metadata"""
    try:
        model_files = list(MODEL_DIR.glob("*.joblib")) + list(MODEL_DIR.glob("*.pkl"))
        
        if not model_files:
            # Try alternate locations
            alt_dir = Path(__file__).parent.parent / "models" / "final"
            if not alt_dir.exists():
                alt_dir = Path("models/final")
            model_files = list(alt_dir.glob("*.joblib")) + list(alt_dir.glob("*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {MODEL_DIR} or alternate locations. Please ensure model files (.joblib or .pkl) exist.")
        
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        model_data = joblib.load(latest_model)
        
        # Extract the actual model object
        if 'stacking' in model_data and hasattr(model_data['stacking'], 'predict'):
            model = model_data['stacking']
        elif 'model' in model_data and isinstance(model_data['model'], dict):
            if 'stacking' in model_data['model']:
                model = model_data['model']['stacking']
            else:
                model = model_data['model']
        elif 'mlp' in model_data:
            model = model_data['mlp']
        else:
            raise ValueError("Could not find model object in loaded data")
        
        feature_names = model_data.get('feature_names', [])
        threshold = model_data.get('threshold', model_data.get('optimal_threshold', 0.5))
        
        accuracy = model_data.get("test_accuracy", model_data.get("accuracy", 0.9429))
        precision = model_data.get("test_precision", model_data.get("precision", 0.9403))
        recall = model_data.get("test_recall", model_data.get("recall", 0.9800))
        f1 = model_data.get("test_f1", model_data.get("f1_score", 0.9597))
        auc = model_data.get("test_auc", model_data.get("auc", 0.9711))
        
        metadata = {
            "filename": latest_model.name,
            "model_type": type(model).__name__,
            "loaded_at": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_score": float(auc),
            "threshold": float(threshold),
            "n_features": len(feature_names),
            "training_samples": model_data.get("training_samples", model_data.get("train_samples", 0))
        }
        
        return model, metadata, feature_names, threshold
        
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def get_confidence_level(probability: float) -> str:
    """Determine confidence level from probability"""
    if probability >= 0.9 or probability <= 0.1:
        return "Very High"
    elif probability >= 0.75 or probability <= 0.25:
        return "High"
    elif probability >= 0.6 or probability <= 0.4:
        return "Moderate"
    else:
        return "Low"

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
    except:
        return [], []

def calculate_shap_values(model, X: pd.DataFrame, max_samples: int = 100):
    """Calculate SHAP explanations"""
    try:
        X_explain = X.iloc[:max_samples] if len(X) > max_samples else X
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        except:
            background = shap.sample(X_explain, min(20, len(X_explain)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        return shap_values
    except Exception as e:
        return None

# ===========================
# STARTUP EVENT
# ===========================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model, metadata, feature_names, threshold = load_model()
        model_cache["model"] = model
        model_cache["metadata"] = metadata
        model_cache["feature_names"] = feature_names
        model_cache["threshold"] = threshold
        model_cache["loaded_at"] = datetime.now().isoformat()
        print(f"[OK] Model loaded successfully: {metadata['filename']}")
    except Exception as e:
        print(f"WARNING:  Warning: Could not load model on startup: {str(e)}")

# ===========================
# API ENDPOINTS
# ===========================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "HEA API - Habitable Exoplanet Analysis",
        "version": "1.0.0",
        "description": "Machine Learning API for exoplanet detection and habitability analysis",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/model/info",
        "endpoints": {
            "prediction": "/predict",
            "habitability": "/habitability",
            "batch": "/predict/batch",
            "explanation": "/predict/explain"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check API health and model status
    
    Returns the current status of the API and whether the ML model is loaded.
    """
    return HealthResponse(
        status="healthy" if model_cache["model"] is not None else "model_not_loaded",
        model_loaded=model_cache["model"] is not None,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get detailed model information and performance metrics
    
    Returns comprehensive information about the loaded ML model including
    accuracy, precision, recall, and other performance metrics.
    """
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = model_cache["metadata"]
    return ModelInfo(
        model_name=metadata["filename"],
        model_type=metadata["model_type"],
        accuracy=metadata["accuracy"],
        precision=metadata["precision"],
        recall=metadata["recall"],
        f1_score=metadata["f1_score"],
        auc_score=metadata["auc_score"],
        threshold=metadata["threshold"],
        n_features=metadata["n_features"],
        training_samples=metadata["training_samples"],
        loaded_at=metadata["loaded_at"],
        version="1.0.0"
    )

@app.get("/model/features", tags=["Model"])
async def get_model_features():
    """
    Get list of features used by the model
    
    Returns the complete list of feature names that the model expects.
    Useful for understanding what data to provide in predictions.
    """
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": model_cache["feature_names"],
        "total_features": len(model_cache["feature_names"]),
        "description": "List of all features expected by the model"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(input_data: ExoplanetInput):
    """
    Make a single exoplanet prediction
    
    Analyzes a single exoplanet candidate and returns the prediction with confidence.
    
    **Example Request:**
    ```json
    {
        "data": {
            "koi_period": 3.5225,
            "koi_depth": 200.5,
            "koi_duration": 2.5,
            "st_teff": 5778
        }
    }
    ```
    """
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.data])
        
        # Align features
        feature_names = model_cache["feature_names"]
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        df = df[feature_names]
        df = preprocess_data(df)
        
        # Make prediction
        model = model_cache["model"]
        threshold = model_cache["threshold"]
        
        proba = model.predict_proba(df)[:, 1][0]
        prediction = "Exoplanet" if proba >= threshold else "Not Exoplanet"
        confidence = get_confidence_level(proba)
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(proba),
            confidence=confidence,
            threshold=float(threshold),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/explain", response_model=ExplanationResponse, tags=["Predictions"])
async def predict_with_explanation(
    input_data: ExoplanetInput,
    top_features: int = Query(5, description="Number of top features to include in explanation", ge=1, le=20)
):
    """
    Make a prediction with SHAP explanation
    
    Returns prediction along with SHAP values explaining which features
    contributed most to the decision.
    
    **Parameters:**
    - **top_features**: Number of most important features to include (1-20)
    """
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.data])
        
        # Align features
        feature_names = model_cache["feature_names"]
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        df = df[feature_names]
        df = preprocess_data(df)
        
        # Make prediction
        model = model_cache["model"]
        threshold = model_cache["threshold"]
        
        proba = model.predict_proba(df)[:, 1][0]
        prediction = "Exoplanet" if proba >= threshold else "Not Exoplanet"
        confidence = get_confidence_level(proba)
        
        # Calculate SHAP values
        shap_values = calculate_shap_values(model, df, max_samples=1)
        
        shap_explanation = None
        if shap_values is not None:
            row_shap = np.array(shap_values[0]).flatten()
            shap_importance = np.abs(row_shap)
            top_indices = np.argsort(shap_importance)[-top_features:][::-1]
            
            top_features_list = []
            for idx in top_indices:
                idx_int = int(idx)
                top_features_list.append({
                    "feature": feature_names[idx_int],
                    "value": float(df.iloc[0, idx_int]),
                    "shap_value": float(row_shap[idx_int]),
                    "importance": float(shap_importance[idx_int]),
                    "direction": "Supports Exoplanet" if row_shap[idx_int] > 0 else "Against Exoplanet"
                })
            
            shap_explanation = {
                "top_features": top_features_list,
                "base_value": 0.0,
                "description": "SHAP values show how much each feature pushes the prediction"
            }
        
        # Get base model votes
        base_votes = None
        estimator_names, estimators = get_base_estimators(model)
        if estimators:
            base_votes = {}
            for name, estimator in zip(estimator_names, estimators):
                try:
                    if hasattr(estimator, 'predict_proba'):
                        est_proba = estimator.predict_proba(df)[:, 1][0]
                        base_votes[name] = float(est_proba)
                except:
                    pass
        
        return ExplanationResponse(
            prediction=prediction,
            probability=float(proba),
            confidence=confidence,
            shap_explanation=shap_explanation,
            base_model_votes=base_votes,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/habitability", response_model=HabitabilityResponse, tags=["Habitability"])
async def calculate_habitability(input_data: HabitabilityInput):
    """
    Calculate habitability score for an exoplanet candidate
    
    Evaluates the habitability potential of an exoplanet based on physical
    characteristics such as radius, temperature, stellar properties, and orbital parameters.
    
    **Scoring Components:**
    - **Radius Score**: How Earth-like is the planet size? (0.8-1.2 R⊕ is optimal)
    - **Temperature Score**: Is temperature suitable for liquid water? (260-320K optimal)
    - **Insolation Score**: Does it receive Earth-like stellar flux? (0.35-1.4 optimal)
    - **Stellar Score**: Is the host star suitable? (G/K-type stars preferred)
    - **Orbital Score**: Is the orbital period in habitable range?
    - **ESI**: Earth Similarity Index (0-1 scale)
    
    **Habitability Classes:**
    - **HIGH**: Score ≥ 0.8 - Very promising candidate
    - **MODERATE**: Score 0.6-0.8 - Potentially habitable
    - **LOW**: Score 0.4-0.6 - Marginal habitability
    - **VERY_LOW**: Score < 0.4 - Unlikely habitable
    
    **Example Request:**
    ```json
    {
        "radius": 1.0,
        "temp": 288.0,
        "insolation": 1.0,
        "stellar_temp": 5778.0,
        "period": 365.25,
        "disposition": "CONFIRMED"
    }
    ```
    
    **Note:** All parameters are optional. Scores are calculated based on available data.
    """
    try:
        # Import HabitabilityScorer
        import sys
        from pathlib import Path
        
        # Add src to path if not already there
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from utils.habitability_scorer import HabitabilityScorer
        
        # Create scorer instance
        scorer = HabitabilityScorer(scoring_mode='balanced')
        
        # Calculate scores
        result = scorer.calculate_habitability_score(
            radius=input_data.radius,
            temp=input_data.temp,
            insolation=input_data.insolation,
            stellar_temp=input_data.stellar_temp,
            period=input_data.period,
            disposition=input_data.disposition
        )
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_type(value):
            if pd.isna(value) or value is None:
                return None
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            return value
        
        return HabitabilityResponse(
            habitability_score=convert_to_python_type(result.get('habitability_score', 0.0)),
            habitability_class=result.get('habitability_class', 'UNKNOWN'),
            radius_score=convert_to_python_type(result.get('radius_score')),
            temperature_score=convert_to_python_type(result.get('temperature_score')),
            insolation_score=convert_to_python_type(result.get('insolation_score')),
            stellar_score=convert_to_python_type(result.get('stellar_score')),
            orbital_score=convert_to_python_type(result.get('orbital_score')),
            esi=convert_to_python_type(result.get('esi')),
            timestamp=datetime.now().isoformat()
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Habitability scorer not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Habitability calculation error: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(input_data: BatchExoplanetInput):
    """
    Make predictions for multiple candidates
    
    Analyzes multiple exoplanet candidates in a single request.
    More efficient than making multiple single predictions.
    
    **Example Request:**
    ```json
    {
        "data": [
            {"koi_period": 3.5225, "koi_depth": 200.5},
            {"koi_period": 4.8921, "koi_depth": 180.3}
        ]
    }
    ```
    """
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Align features
        feature_names = model_cache["feature_names"]
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        df = df[feature_names]
        df = preprocess_data(df)
        
        # Make predictions
        model = model_cache["model"]
        threshold = model_cache["threshold"]
        
        probas = model.predict_proba(df)[:, 1]
        predictions = ["Exoplanet" if p >= threshold else "Not Exoplanet" for p in probas]
        
        # Create response list
        prediction_list = []
        for pred, proba in zip(predictions, probas):
            prediction_list.append(PredictionResponse(
                prediction=pred,
                probability=float(proba),
                confidence=get_confidence_level(proba),
                threshold=float(threshold),
                timestamp=datetime.now().isoformat()
            ))
        
        # Calculate summary
        n_exoplanets = sum(1 for p in predictions if p == "Exoplanet")
        summary = {
            "total_candidates": len(predictions),
            "predicted_exoplanets": n_exoplanets,
            "predicted_non_exoplanets": len(predictions) - n_exoplanets,
            "avg_probability": float(np.mean(probas)),
            "max_probability": float(np.max(probas)),
            "min_probability": float(np.min(probas))
        }
        
        return BatchPredictionResponse(
            predictions=prediction_list,
            summary=summary,
            total_processed=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/batch/upload", tags=["Predictions"])
async def predict_from_csv(
    file: UploadFile = File(..., description="CSV file with exoplanet candidate data")
):
    """
    Upload CSV file for batch predictions
    
    Upload a CSV file containing multiple exoplanet candidates.
    The API will process all rows and return predictions.
    
    **CSV Format:**
    - First row should contain column headers
    - Each subsequent row represents one candidate
    - Missing features will be filled with zeros
    
    **Returns:** JSON with predictions and downloadable results
    """
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Store original data
        df_original = df.copy()
        
        # Align features
        feature_names = model_cache["feature_names"]
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        df_processed = df[feature_names].copy()
        df_processed = preprocess_data(df_processed)
        
        # Make predictions
        model = model_cache["model"]
        threshold = model_cache["threshold"]
        
        probas = model.predict_proba(df_processed)[:, 1]
        predictions = ["Exoplanet" if p >= threshold else "Not Exoplanet" for p in probas]
        confidences = [get_confidence_level(p) for p in probas]
        
        # Add predictions to original dataframe
        df_original['prediction'] = predictions
        df_original['probability'] = probas
        df_original['confidence'] = confidences
        
        # Calculate summary
        n_exoplanets = sum(1 for p in predictions if p == "Exoplanet")
        summary = {
            "total_candidates": len(predictions),
            "predicted_exoplanets": n_exoplanets,
            "predicted_non_exoplanets": len(predictions) - n_exoplanets,
            "avg_probability": float(np.mean(probas)),
            "max_probability": float(np.max(probas)),
            "min_probability": float(np.min(probas)),
            "exoplanet_percentage": f"{(n_exoplanets/len(predictions)*100):.1f}%"
        }
        
        # Convert to JSON-friendly format
        results = df_original.to_dict(orient='records')
        
        return {
            "status": "success",
            "filename": file.filename,
            "summary": summary,
            "results": results,
            "total_processed": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV processing error: {str(e)}")

@app.get("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload the ML model from disk
    
    Useful if the model has been updated and needs to be reloaded
    without restarting the API.
    """
    try:
        model, metadata, feature_names, threshold = load_model()
        model_cache["model"] = model
        model_cache["metadata"] = metadata
        model_cache["feature_names"] = feature_names
        model_cache["threshold"] = threshold
        model_cache["loaded_at"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_name": metadata["filename"],
            "loaded_at": metadata["loaded_at"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

# ===========================
# ERROR HANDLERS
# ===========================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The endpoint {request.url.path} does not exist",
            "documentation": "/docs"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "support": "Please check the logs or contact support"
        }
    )

# ===========================
# RUN SERVER
# ===========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
