# HEA FastAPI Backend

**Habitable Exoplanet Analysis - REST API**  
NASA Space Apps Challenge 2025

## Overview

High-performance REST API for exoplanet detection using machine learning. Provides programmatic access to the HEA ML model for researchers and developers.

### Features

- Single and batch prediction endpoints
- SHAP-based explainability for predictions
- Model performance metrics and feature information
- CSV file upload for bulk processing
- Auto-generated OpenAPI documentation (Swagger UI & ReDoc)
- CORS support for web integration
- Production-ready error handling and validation
- Async endpoint support with FastAPI

### Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94.29% |
| Precision | 94.03% |
| Recall | 98.00% |
| F1-Score | 95.97% |
| AUC | 0.97 |

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Trained model files in `../models/catalog_models/`

### Installation & Launch

#### Option 1: PowerShell Launcher (Recommended)

```powershell
.\launch_backend.ps1
```

This script will:

1. Create a virtual environment
2. Install all dependencies
3. Start the server with auto-reload

#### Option 2: Manual Setup

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the API

Once running, access:

- **Interactive Docs (Swagger)**: <http://localhost:8000/docs>
- **Alternative Docs (ReDoc)**: <http://localhost:8000/redoc>
- **API Root**: <http://localhost:8000/>

## API Endpoints

### General

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |

### Model

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/model/info` | Model metrics & info |
| `GET` | `/model/features` | List of model features |
| `GET` | `/model/reload` | Reload model from disk |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/explain` | Prediction with SHAP |
| `POST` | `/predict/batch` | Batch predictions |
| `POST` | `/predict/batch/upload` | CSV upload |

## Usage Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "koi_period": 3.5225,
      "koi_depth": 200.5,
      "st_teff": 5778
    }
  }'

# Upload CSV
curl -X POST "http://localhost:8000/predict/batch/upload" \
  -F "file=@sample_data.csv"
```

### Python

```python
import requests

# Single prediction
data = {
    "data": {
        "koi_period": 3.5225,
        "koi_depth": 200.5,
        "st_teff": 5778
    }
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())

# Batch prediction
batch_data = {
    "data": [
        {"koi_period": 3.5225, "koi_depth": 200.5},
        {"koi_period": 4.8921, "koi_depth": 180.3}
    ]
}
response = requests.post("http://localhost:8000/predict/batch", json=batch_data)
print(response.json())
```

### PowerShell

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Single prediction
$body = @{
    data = @{
        koi_period = 3.5225
        koi_depth = 200.5
    }
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Testing

Run the test suite:

```bash
python test_api.py
```

Tests include:

- Health endpoint
- Model info
- Features list
- Single prediction
- Prediction with explanation
- Batch prediction

## Documentation

Complete API documentation:

- **API_DOCUMENTATION.md** - Full documentation with examples
- **Swagger UI** - Interactive testing at <http://localhost:8000/docs>
- **ReDoc** - Clean documentation at <http://localhost:8000/redoc>

## Architecture

```text
backend/
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── launch_backend.ps1         # PowerShell launcher
├── test_api.py               # Test suite
├── API_DOCUMENTATION.md      # Full documentation
└── README.md                 # This file

../models/
└── catalog_models/           # Trained models
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
MODEL_PATH=../models/catalog_models
API_PORT=8000
API_HOST=0.0.0.0
LOG_LEVEL=info
```

### CORS Configuration

CORS is enabled by default for all origins. To restrict:

```python
# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Your specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🚢 Deployment

### Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ../models ../models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t hea-api .
docker run -p 8000:8000 hea-api
```

### Cloud Deployment

#### Azure

```bash
az webapp up --runtime PYTHON:3.11 --name hea-api
```

#### AWS

```bash
eb init -p python-3.11 hea-api
eb create hea-api-env
```

#### Google Cloud

```bash
gcloud app deploy
```

## Troubleshooting

### Model Not Loading

**Problem:** API returns 503 "Model not loaded"

**Solution:**

1. Check model files exist: `ls ../models/catalog_models/`
2. Verify model format (`.joblib` or `.pkl`)
3. Check file permissions
4. Review server logs

### Port Already in Use

**Problem:** Port 8000 is already in use

**Solution:**

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or use a different port
uvicorn main:app --port 8001
```

### Import Errors

**Problem:** Module not found errors

**Solution:**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Integration with Web App

Run API and Streamlit web app simultaneously:

```powershell
# Terminal 1: API Backend
cd backend
.\launch_backend.ps1

# Terminal 2: Streamlit Web App
cd ..
.\launch_webapp.ps1
```

Access:

- **API**: <http://localhost:8000>
- **Web App**: <http://localhost:8501>

## Performance

- **Startup Time**: ~2-3 seconds
- **Single Prediction**: ~10-20ms
- **Batch (100 items)**: ~100-200ms
- **SHAP Explanation**: ~500ms-1s

### Optimization Tips

- Use batch endpoints for multiple predictions
- Only request explanations when needed
- Cache results for identical inputs
- Use async clients for parallel requests

## Security

### Production Checklist

- Use HTTPS (TLS/SSL certificates)
- Implement API key authentication
- Rate limiting (e.g., with slowapi)
- Input validation (built-in with Pydantic)
- Error message sanitization
- CORS restriction to known origins
- Regular dependency updates

### Adding API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Apply to endpoints
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_single(...):
    ...
```

## License

MIT License - NASA Space Apps Challenge 2025

## Support

For issues or questions:

1. Check API_DOCUMENTATION.md
2. Review error messages in response
3. Check server logs
4. Open an issue on GitHub

## Roadmap

- WebSocket support for real-time predictions
- GraphQL API endpoint
- Authentication and API keys
- Rate limiting
- Caching layer (Redis)
- Metrics and monitoring (Prometheus)
- Load testing results
- Multi-model support
- A/B testing framework
