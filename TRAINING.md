# Model Training Guide

## Overview

The Fleet Maintenance Cost Predictor uses LSTM neural networks to predict maintenance costs. Before using the API, you need to train the models.

## Prerequisites

- Training data CSV file in `data/` directory
- Python 3.12+ or Docker

## Training Methods

### Method 1: Docker (Recommended)

**Train models inside the running container:**

```bash
# Start the backend if not running
docker-compose up -d

# Run training inside container
docker exec -it fleet-backend python train_cost_models.py

# Or on Windows PowerShell:
docker exec -it fleet-backend python train_cost_models.py
```

### Method 2: Local Python

**Train on your local machine:**

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training
python train_cost_models.py
```

### Method 3: During Docker Build

**Uncomment the training step in Dockerfile:**

Edit `Dockerfile` and add before the CMD line:

```dockerfile
# Train models during build (optional - increases build time)
RUN python train_cost_models.py
```

Then rebuild:
```bash
docker-compose up -d --build
```

## What Happens During Training

The script will:

1. **Load training data** from `data/Bus_Maintenance_TimeSeries_2023-2025.csv`
2. **Train 4 LSTM models** (one for each subsystem):
   - HV_Battery_cost_model
   - Traction_Motor_cost_model
   - HVAC_Roof_cost_model
   - Air_Compressor_cost_model
3. **Save models** to `models/` directory
4. **Save scalers and metadata** for each model

## Expected Output

```
==========================================
Training Cost Prediction Models
==========================================

Loading training data...
Data loaded: 96 records

Training subsystem: HV_Battery
  Processing 90 sequences...
  Training LSTM model (30 epochs)...
  Epoch 1/30 - Loss: 0.234
  Epoch 2/30 - Loss: 0.189
  ...
  Model saved to models/HV_Battery_cost_model/

Training subsystem: Traction_Motor
  ...

Training subsystem: HVAC_Roof
  ...

Training subsystem: Air_Compressor
  ...

==========================================
Training Complete!
==========================================
All 4 models trained successfully.
Models saved to: models/
```

## Training Time

- **Prototype mode** (default): ~2-3 minutes
- **Full dataset**: ~10-15 minutes

## Verifying Trained Models

After training, check that models exist:

```bash
# Local
ls -lh models/

# Docker
docker exec fleet-backend ls -lh /app/models/
```

You should see 4 directories:
```
models/
├── Air_Compressor_cost_model/
│   ├── model.h5
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   └── metadata.json
├── HV_Battery_cost_model/
├── HVAC_Roof_cost_model/
└── Traction_Motor_cost_model/
```

## Troubleshooting

### "Training data not found"
```bash
# Ensure CSV file exists
ls data/Bus_Maintenance_TimeSeries_2023-2025.csv

# If missing, copy it to data/
cp /path/to/your/csv/file data/
```

### "Module not found" errors
```bash
# Ensure dependencies are installed
pip install -r requirements.txt
```

### Memory issues
The LSTM models require ~2GB RAM. If you have limited memory:
- Close other applications
- Use prototype mode (enabled by default in config.py)
- Train one model at a time

### Models not loading in API
```bash
# Restart the backend after training
docker-compose restart

# Or if running locally
# Stop and start the API server
```

## Re-training Models

To re-train models (e.g., with new data):

```bash
# Backup existing models (optional)
cp -r models/ models_backup/

# Run training again
python train_cost_models.py

# The script will overwrite existing models
```

## Production Deployment

For production:

1. **Train models locally** before deploying
2. **Include trained models** in your Docker image:
   ```dockerfile
   # In Dockerfile, after COPY . .
   RUN python train_cost_models.py
   ```
3. **Or mount models as volume**:
   ```yaml
   # In docker-compose.yml
   volumes:
     - ./models:/app/models:ro  # Read-only
   ```

## Next Steps

After training:
1. ✅ Models are ready to use
2. ✅ Start the API server
3. ✅ Test predictions via API or frontend

```bash
# Start backend
docker-compose up -d

# Test health endpoint
curl http://localhost:8000/api/health

# You should see models_loaded: ["HV_Battery", ...]
```

## Support

If training fails:
- Check the logs for specific error messages
- Ensure training data CSV is valid
- Verify sufficient disk space (~100MB for models)
- Check Python/TensorFlow compatibility
