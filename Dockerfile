# Use Python 3.12 slim image for smaller size
FROM python:3.12.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set PYTHONPATH to include project root
ENV PYTHONPATH=/app

# Train models during build (so they're ready when container starts)
# Using ML models (Random Forest/XGBoost) instead of LSTM
RUN PYTHONPATH=/app python train_cost_models.py

# Expose port (Render/Railway/etc will override with $PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

# Run the FastAPI server
CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}
