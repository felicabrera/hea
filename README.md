# HEA - Habitable Exoplanet Analyzer

**NASA Space Apps Challenge 2025**

An AI-powered machine learning system for exoplanet detection and habitability assessment, featuring an interactive web interface and REST API backend.

## Overview

HEA (Habitable Exoplanet Analyzer) combines advanced machine learning with explainable AI to detect exoplanet candidates and evaluate their potential habitability. The system achieves 94.29% accuracy using a stacking ensemble classifier and provides detailed SHAP-based explanations for each prediction.

### Key Features

- **Stacking Ensemble Model**: Random Forest, Gradient Boosting, XGBoost, and LightGBM
- **SHAP Explainability**: Model agreement visualization and feature importance analysis
- **Habitability Assessment**: Multi-component scoring system for planetary habitability
- **Interactive Web Interface**: Modern Streamlit application with real-time predictions
- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Multi-Mission Support**: Analyzes data from Kepler, TESS, and K2 missions
- **Batch Processing**: CSV upload for bulk candidate analysis

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94.29% |
| Precision | 94.03% |
| Recall | 98.00% |
| F1-Score | 95.97% |
| AUC | 0.97 |

## Quick Start

### Option 1: Docker (Recommended)

**Easiest way to run HEA - no Python installation needed!**

```powershell
# Windows
.\docker-start.ps1

# Linux/Mac
chmod +x docker-start.sh
./docker-start.sh
```

Services will be available at:
- **Webapp**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

See [Docker Setup Guide](docs/DOCKER_SETUP.md) for detailed configuration.

### Option 2: Python Local

**Prerequisites**: Python 3.8 or higher

#### Web Application

```powershell
.\launch_webapp.ps1
```

Access at: http://localhost:8502

#### REST API Backend

```powershell
cd backend
.\launch_backend.ps1
```

API Documentation: http://localhost:8000/docs

## Project Structure

```
hea/
├── app_modern.py           # Main Streamlit application
├── backend/                # FastAPI REST API
│   ├── main.py
│   └── requirements.txt
├── config/                 # Configuration files
├── data/
│   └── catalogs/          # Kepler, TESS, K2 datasets
├── models/
│   └── catalog_models/    # Trained ML models
├── scripts/
│   └── analyze_habitability.py  # Habitability analysis
├── src/
│   └── utils/
│       └── habitability_scorer.py
└── requirements.txt
```

## Installation

### Automated Setup (Recommended)

The launcher scripts automatically handle environment setup and dependency installation.

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app_modern.py --server.port 8502
```

## Features

### Detection System

- **Single Prediction Mode**: Analyze individual exoplanet candidates with detailed parameter inputs
- **Batch Upload Mode**: Process multiple candidates via CSV file upload
- **Adjustable Threshold**: Fine-tune classification sensitivity with custom decision thresholds
- **Model Insights**: View ensemble model agreement and base estimator contributions

### Explainability

- **SHAP Analysis**: Waterfall plots and force plots for prediction explanations
- **Feature Impact**: Top contributing features with relative importance
- **Model Agreement**: Visualization of base estimator consensus

### Habitability Assessment

- **Component Scoring**: Radius, temperature, stellar, and orbital parameter analysis
- **Multi-Mission Rankings**: Top 100 most habitable candidates from combined datasets
- **Physical Properties**: Display of available planetary and stellar characteristics
- **Detailed Breakdowns**: Expandable candidate analysis with strengths and weaknesses

### API Capabilities

- Single and batch predictions
- SHAP explanations
- Model metadata and feature lists
- CSV file upload
- Health monitoring

## Data Sources

The system analyzes candidates from:

- **Kepler Mission**: 9,564 Kepler Objects of Interest (KOI)
- **TESS Mission**: 7,703 TESS Objects of Interest (TOI)
- **K2 Mission**: 4,004 candidate planets

Data sourced from NASA Exoplanet Archive.

## Configuration

Configuration files in `config/`:

- `config.yaml`: Project settings and data paths
- `model_config.yaml`: Model hyperparameters and architecture
- `tuning_config.yaml`: Hyperparameter tuning ranges

## Requirements

Key dependencies:

- streamlit >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- shap >= 0.42.0
- plotly >= 5.14.0
- fastapi >= 0.100.0
- uvicorn >= 0.23.0

See `requirements.txt` for complete list.

## API Usage

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"data": {"koi_period": 3.5, "koi_depth": 200.5, "st_teff": 5778}}
)
print(response.json())
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": {"koi_period": 3.5, "koi_depth": 200.5}}'
```

## Documentation

- **Backend API**: See `backend/README.md` for detailed API documentation
- **API Reference**: Interactive docs at http://localhost:8000/docs
- **Project Docs**: Additional documentation in `docs/` directory

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- NASA Exoplanet Archive
- NASA Space Apps Challenge 2025
- Kepler, TESS, and K2 mission teams
