"""
Configuration file for Fleet Predictive Maintenance System - Backend
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Subsystems to monitor
SUBSYSTEMS = {
    'HV_Battery': {
        'name': 'HV Battery Packs',
        'sensors': ['voltage', 'current', 'temperature', 'soc'],
        'base_cost': 510,
        'degradation_rate': 0.15,  # 15% per year
        'failure_threshold': 0.7
    },
    'Traction_Motor': {
        'name': 'Traction Motor',
        'sensors': ['vibration', 'temperature', 'rpm', 'current'],
        'base_cost': 105,
        'degradation_rate': 0.08,
        'failure_threshold': 0.75
    },
    'HVAC_Roof': {
        'name': 'Roof HVAC Unit',
        'sensors': ['temperature', 'pressure', 'airflow', 'power_consumption'],
        'base_cost': 90,
        'degradation_rate': 0.12,
        'failure_threshold': 0.65
    },
    'Air_Compressor': {
        'name': 'Air Compressor (Braking)',
        'sensors': ['pressure', 'temperature', 'vibration', 'duty_cycle'],
        'base_cost': 50,
        'degradation_rate': 0.10,
        'failure_threshold': 0.70
    }
}

# Time series configuration
SAMPLING_INTERVAL_SECONDS = 15
HISTORICAL_MONTHS = 6
SEQUENCE_LENGTH = 240
PREDICTION_HORIZON = 30

# Prototype mode
PROTOTYPE_MODE = True
PROTOTYPE_SAMPLES = 2000

# LSTM Model configuration
LSTM_CONFIG = {
    'units': [128, 64, 32],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 30,
    'validation_split': 0.2
}

# API configuration
API_HOST = '0.0.0.0'
API_PORT = 8000

# CORS configuration - Update this with your frontend URL in production
CORS_ORIGINS = [
    "http://localhost",           # Production frontend (port 80)
    "http://localhost:80",        # Production frontend (explicit port)
    "http://localhost:3000",      # Development frontend
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1",           # Localhost alternative
    "http://127.0.0.1:80",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*",                          # Allow all origins (for development only)
]

# Bus fleet information
FLEET_INFO = {
    'bus_id': 'BUS_001',
    'make': 'New Flyer',
    'model': 'Xcelsior NG',
    'age_months': 66,
    'mileage': 220000
}
