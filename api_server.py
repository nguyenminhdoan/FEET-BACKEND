"""
FastAPI server for cost prediction
Provides REST API for fleet maintenance cost predictions
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
import json
import os

from config import MODEL_DIR, CORS_ORIGINS

app = FastAPI(
    title="Fleet Maintenance Cost Predictor API",
    description="LSTM-based cost forecasting system for fleet maintenance",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store loaded models in memory
loaded_models = {}


class PredictionRequest(BaseModel):
    subsystem: str = Field(..., description="Subsystem to predict costs for")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    age_months: int = Field(..., description="Current bus age in months")
    mileage_km: int = Field(..., description="Current bus mileage in kilometers")
    bus_make: str = Field(..., description="Bus manufacturer")
    bus_model: str = Field(..., description="Bus model")
    year: Optional[int] = Field(None, description="Bus manufacturing year")
    length_ft: Optional[int] = Field(None, description="Bus length in feet")
    propulsion_type: Optional[str] = Field(None, description="Bus propulsion type")

    class Config:
        json_schema_extra = {
            "example": {
                "subsystem": "HV_Battery",
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
                "age_months": 66,
                "mileage_km": 220000,
                "bus_make": "New Flyer",
                "bus_model": "Xcelsior NG",
                "year": 2023,
                "length_ft": 40,
                "propulsion_type": "CNG"
            }
        }


class PredictionData(BaseModel):
    date: str
    cost: float
    age_months: int
    mileage_km: int


class SummaryData(BaseModel):
    average_cost: float
    total_cost: float
    min_cost: float
    max_cost: float
    cost_trend: float


class PredictionResponse(BaseModel):
    subsystem: str
    predictions: List[PredictionData]
    summary: SummaryData


class SubsystemInfo(BaseModel):
    id: str
    name: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    available_subsystems: List[str]


def load_cost_model(subsystem_name: str):
    """Load a trained cost prediction model"""
    if subsystem_name in loaded_models:
        return loaded_models[subsystem_name]

    model_dir = os.path.join(MODEL_DIR, f'{subsystem_name}_cost_model')

    if not os.path.exists(model_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Model for {subsystem_name} not found. Please train the model first."
        )

    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model metadata for {subsystem_name} not found"
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load model
    model_path = os.path.join(model_dir, 'model.h5')
    model = keras.models.load_model(model_path, compile=False)

    # Load scalers
    scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))

    model_data = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'metadata': metadata
    }

    loaded_models[subsystem_name] = model_data
    return model_data


def predict_future_costs(subsystem_name: str, start_date_str: str, end_date_str: str,
                         initial_age_months: int, initial_mileage_km: int,
                         bus_make: str = "New Flyer", bus_model: str = "Xcelsior NG",
                         year: int = None, length_ft: int = None, propulsion_type: str = None):
    """Predict costs for a date range"""

    # Parse dates
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD"
        )

    # Calculate months
    months_diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    if months_diff < 1:
        raise HTTPException(
            status_code=400,
            detail="End date must be at least 1 month after start date"
        )

    if months_diff > 24:
        raise HTTPException(
            status_code=400,
            detail="Maximum prediction range is 24 months"
        )

    # Try to load data from multiple sources
    # High-level categories (1. Propulsion, 10. HVAC, etc.) use old CSV
    # Detailed subsystems use YRT CSV
    subsystem_df = None

    # Try YRT data first (detailed subsystems)
    csv_path = 'data/YRT_FZ_sample_data.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        subsystem_df = df[df['Subsystem'] == subsystem_name].copy()

    # If not found, try old data (high-level categories)
    if subsystem_df is None or len(subsystem_df) == 0:
        csv_path = 'data/Bus_Maintenance_TimeSeries_2023-2025.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            subsystem_df = df[df['Subsystem'] == subsystem_name].copy()

    # If still not found, error
    if subsystem_df is None or len(subsystem_df) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data for subsystem: {subsystem_name}"
        )

    subsystem_df = subsystem_df.sort_values('Date').reset_index(drop=True)

    # Add time features
    subsystem_df['month'] = subsystem_df['Date'].dt.month
    subsystem_df['days_since_start'] = (subsystem_df['Date'] - subsystem_df['Date'].min()).dt.days

    # Load model
    model_data = load_cost_model(subsystem_name)
    model = model_data['model']
    scaler_X = model_data['scaler_X']
    scaler_y = model_data['scaler_y']
    metadata = model_data['metadata']

    # Get feature columns from metadata
    feature_cols = metadata.get('feature_cols', [])

    # Create future features
    last_6_months = subsystem_df.tail(6).copy()
    predictions = []

    for month_idx in range(months_diff + 1):
        # Calculate target date
        target_date = start_date + timedelta(days=30.5 * month_idx)

        # Get last row
        last_row = last_6_months.iloc[-1]

        # Use user-provided initial values for the first prediction, then increment
        if month_idx == 0:
            new_age = initial_age_months
            new_mileage = initial_mileage_km
        else:
            new_age = last_row['Age_months'] + 1
            new_mileage = last_row['Mileage_km'] + 10000

        new_month = target_date.month
        new_days = (target_date - subsystem_df.iloc[0]['Date']).days

        # Aging factor
        aging_factor = 1.01
        pm_labour = last_row['PM_Labour_Cost'] * aging_factor
        pm_parts = last_row['PM_Parts_Cost'] * aging_factor
        cm_labour = last_row['CM_Labour_Cost'] * aging_factor
        cm_parts = last_row['CM_Parts_Cost'] * aging_factor

        # Create feature row
        row_data = {
            'Date': target_date,
            'PM_Labour_Cost': pm_labour,
            'PM_Parts_Cost': pm_parts,
            'CM_Labour_Cost': cm_labour,
            'CM_Parts_Cost': cm_parts,
            'Age_months': new_age,
            'Mileage_km': new_mileage,
            'month': new_month,
            'days_since_start': new_days
        }

        new_row = pd.DataFrame([row_data])

        # Update rolling window
        last_6_months = pd.concat([last_6_months, new_row]).tail(6).reset_index(drop=True)

        # Prepare sequence for prediction using the model's expected features
        sequence = last_6_months[feature_cols].values
        n_features = len(feature_cols)
        sequence_scaled = scaler_X.transform(sequence.reshape(-1, n_features)).reshape(1, 6, n_features)

        # Predict
        pred_scaled = model.predict(sequence_scaled, verbose=0)
        pred_cost = scaler_y.inverse_transform(pred_scaled)[0][0]

        predictions.append({
            'date': target_date.strftime('%Y-%m-%d'),
            'cost': round(float(pred_cost), 2),
            'age_months': int(new_age),
            'mileage_km': int(new_mileage)
        })

    # Calculate summary
    costs = [p['cost'] for p in predictions]
    summary = {
        'average_cost': round(float(np.mean(costs)), 2),
        'total_cost': round(float(np.sum(costs)), 2),
        'min_cost': round(float(np.min(costs)), 2),
        'max_cost': round(float(np.max(costs)), 2),
        'cost_trend': round(float(costs[-1] - costs[0]), 2)
    }

    return {
        'subsystem': subsystem_name,
        'predictions': predictions,
        'summary': summary
    }


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Fleet Maintenance Cost Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/subsystems", response_model=List[SubsystemInfo])
async def get_subsystems():
    """Get list of available subsystems from trained models"""
    subsystems = []

    if not os.path.exists(MODEL_DIR):
        return subsystems

    # Scan the models directory for trained subsystem models
    for item in os.listdir(MODEL_DIR):
        model_path = os.path.join(MODEL_DIR, item)

        # Check if it's a directory and contains a metadata.json file
        if os.path.isdir(model_path):
            metadata_file = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Extract subsystem name from metadata or folder name
                    subsystem_name = metadata.get('subsystem', item.replace('_cost_model', ''))

                    # Filter out subsystems with tiny datasets (< 10 samples)
                    # These are marked with 🚨 in training output and are very unreliable
                    original_records = metadata.get('original_records', 0)
                    if original_records < 10:
                        print(f"Skipping {subsystem_name} - only {original_records} samples (unreliable)")
                        continue

                    # Create ID by replacing spaces with underscores and removing special chars
                    subsystem_id = subsystem_name.replace(' ', '_').replace('/', '_').replace('&', 'and')
                    subsystem_id = ''.join(c for c in subsystem_id if c.isalnum() or c == '_')

                    subsystems.append({
                        "id": subsystem_id,
                        "name": subsystem_name
                    })
                except Exception as e:
                    print(f"Error reading metadata for {item}: {e}")
                    continue

    # Sort subsystems by name for better UX
    subsystems = sorted(subsystems, key=lambda x: x['name'])

    return subsystems


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_costs(request: PredictionRequest):
    """Predict costs for a subsystem over a date range"""
    try:
        # Convert subsystem ID to name
        subsystems = await get_subsystems()
        subsystem_map = {s["id"]: s["name"] for s in subsystems}
        subsystem_name = subsystem_map.get(request.subsystem, request.subsystem)

        result = predict_future_costs(
            subsystem_name=subsystem_name,
            start_date_str=request.start_date,
            end_date_str=request.end_date,
            initial_age_months=request.age_months,
            initial_mileage_km=request.mileage_km,
            bus_make=request.bus_make,
            bus_model=request.bus_model,
            year=request.year,
            length_ft=request.length_ft,
            propulsion_type=request.propulsion_type
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Get available subsystems dynamically
    subsystems = await get_subsystems()
    subsystem_ids = [s["id"] for s in subsystems]

    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "available_subsystems": subsystem_ids
    }


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("Fleet Maintenance Cost Prediction API")
    print("="*60)
    print("\nStarting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
