# Fleet Maintenance Cost Predictor - Backend

This is the backend API server for the Fleet Maintenance Cost Predictor application. It provides REST API endpoints for LSTM-based cost forecasting.

## Tech Stack

- **Framework**: FastAPI
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Python Version**: 3.12

## Project Structure

```
fleet-backend/
├── api_server.py          # Main FastAPI application
├── config.py              # Configuration settings
├── train_cost_models.py   # Model training script (copy from parent)
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── data/                 # Training data directory
│   └── Bus_Maintenance_TimeSeries_2023-2025.csv
├── models/               # Trained models directory
│   ├── HV_Battery_cost_model/
│   ├── Traction_Motor_cost_model/
│   ├── HVAC_Roof_cost_model/
│   └── Air_Compressor_cost_model/
└── README.md
```

## Setup

### Prerequisites

- Python 3.12+
- pip

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy training data (if not already present):
```bash
# The CSV file should be in the data/ directory
# If missing, copy it from your source
cp /path/to/Bus_Maintenance_TimeSeries_2023-2025.csv data/
```

3. Train models (if not already trained):
```bash
# Training script is included in the backend folder
python train_cost_models.py
```

4. Run the server:
```bash
python api_server.py
# or
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Docker Setup

### Quick Start with Docker Compose (Recommended):

**Production Mode:**
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```

**Development Mode (with hot reload):**
```bash
# Windows
start-dev.bat

# Linux/Mac
./start-dev.sh
```

### Manual Docker Commands:

```bash
# Production
docker-compose up -d --build

# Development
docker-compose -f docker-compose.dev.yml up --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Build and run without Docker Compose:

```bash
# Build the image
docker build -t fleet-backend .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  fleet-backend
```

## API Endpoints

### GET `/`
Root endpoint with API information

### GET `/api/health`
Health check endpoint
```json
{
  "status": "healthy",
  "models_loaded": ["HV_Battery"],
  "available_subsystems": ["HV_Battery", "Traction_Motor", "HVAC_Roof", "Air_Compressor"]
}
```

### GET `/api/subsystems`
Get list of available subsystems
```json
[
  {"id": "HV_Battery", "name": "HV Battery Packs"},
  {"id": "Traction_Motor", "name": "Traction Motor"}
]
```

### POST `/api/predict`
Predict maintenance costs

**Request:**
```json
{
  "subsystem": "HV_Battery",
  "start_date": "2026-01-01",
  "end_date": "2026-12-31"
}
```

**Response:**
```json
{
  "subsystem": "HV_Battery",
  "predictions": [
    {
      "date": "2026-01-01",
      "cost": 523.45,
      "age_months": 67,
      "mileage_km": 230000
    }
  ],
  "summary": {
    "average_cost": 525.30,
    "total_cost": 6303.60,
    "min_cost": 520.10,
    "max_cost": 530.50,
    "cost_trend": 10.40
  }
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

Edit `config.py` to modify:
- CORS origins (for frontend integration)
- Model paths
- LSTM configuration
- Subsystem definitions

## Development

### Running in development mode with auto-reload:
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### Testing the API:
```bash
# Using curl
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subsystem": "HV_Battery",
    "start_date": "2026-01-01",
    "end_date": "2026-06-30"
  }'
```

## Environment Variables

- `PORT`: Server port (default: 8000)
- `PYTHONPATH`: Should be set to `/app` for Docker

## Standalone Deployment

This backend can run independently and serve any frontend application:

1. **Start the backend:**
   ```bash
   ./start.sh  # or start.bat on Windows
   ```

2. **Configure CORS** in `config.py` to allow your frontend domain:
   ```python
   CORS_ORIGINS = [
       "http://localhost:3000",
       "https://your-frontend-domain.com",
   ]
   ```

3. **Access the API:**
   - Health check: http://localhost:8000/api/health
   - API docs: http://localhost:8000/docs

## Working with the Frontend

If you have the fleet-frontend project:

1. **Start backend first:**
   ```bash
   cd fleet-backend
   ./start.sh
   ```

2. **Start frontend separately:**
   ```bash
   cd ../fleet-frontend
   ./start.sh
   ```

3. **Access:**
   - Frontend: http://localhost or http://localhost:3000
   - Backend: http://localhost:8000

## Notes

- The API supports CORS for frontend integration
- Models are loaded lazily (on first request)
- Maximum prediction range is 24 months
- Training data should be placed in the `data/` directory
- Trained models should be placed in the `models/` directory
- **This backend is completely independent** - can be moved anywhere and run standalone

## License

MIT
