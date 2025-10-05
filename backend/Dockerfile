# Dockerfile for HEA Backend API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend directory
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY backend/main.py .

# Copy shared directories from project root
COPY models ./models
COPY config ./config
COPY src ./src

# Create necessary directories
RUN mkdir -p logs

# Expose FastAPI default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run the backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
