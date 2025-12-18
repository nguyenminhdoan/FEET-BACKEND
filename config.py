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

# Subsystems to monitor - Based on real YRT BEB data
SUBSYSTEMS = {
    'Propulsion': {
        'name': '1. Propulsion',
        'full_name': '1. Propulsion',
        'base_cost': 196,
        'degradation_rate': 0.08,
        'failure_threshold': 0.75
    },
    'ESS_HV': {
        'name': '2. Energy Storage & High Voltage (ESS & HV)',
        'full_name': '2. Energy Storage & High Voltage (ESS & HV)',
        'base_cost': 1876,
        'degradation_rate': 0.15,
        'failure_threshold': 0.70
    },
    'Low_Voltage_Electrical': {
        'name': '3. Low Voltage Electrical',
        'full_name': '3. Low Voltage Electrical',
        'base_cost': 383,
        'degradation_rate': 0.10,
        'failure_threshold': 0.70
    },
    'Brakes_Pneumatics_ABS': {
        'name': '4. Brakes, Pneumatics and ABS',
        'full_name': '4. Brakes, Pneumatics and ABS',
        'base_cost': 1417,
        'degradation_rate': 0.12,
        'failure_threshold': 0.65
    },
    'Axle_Suspension_Differential': {
        'name': '5. Axle, Suspension & Differential',
        'full_name': '5. Axle, Suspension & Differential',
        'base_cost': 507,
        'degradation_rate': 0.10,
        'failure_threshold': 0.70
    },
    'Steering': {
        'name': '6. Steering',
        'full_name': '6. Steering',
        'base_cost': 286,
        'degradation_rate': 0.08,
        'failure_threshold': 0.75
    },
    'Exterior_Body': {
        'name': '8. Exterior Body',
        'full_name': '8. Exterior Body',
        'base_cost': 400,
        'degradation_rate': 0.05,
        'failure_threshold': 0.80
    },
    'Interior': {
        'name': '9. Interior',
        'full_name': '9. Interior',
        'base_cost': 84,
        'degradation_rate': 0.05,
        'failure_threshold': 0.80
    },
    'HVAC': {
        'name': '10. HVAC',
        'full_name': '10. HVAC',
        'base_cost': 529,
        'degradation_rate': 0.12,
        'failure_threshold': 0.65
    },
    'ITS_TMS': {
        'name': '11. ITS / Technology Management System (TMS)',
        'full_name': '11. ITS / Technology Management System (TMS)',
        'base_cost': 67,
        'degradation_rate': 0.08,
        'failure_threshold': 0.75
    },
    'Doors_Ramps_Accessibility': {
        'name': '12. Doors & Ramps / Accessibility',
        'full_name': '12. Doors & Ramps / Accessibility',
        'base_cost': 193,
        'degradation_rate': 0.10,
        'failure_threshold': 0.70
    },
    'Wheels_Tires': {
        'name': '13. Wheels and Tires',
        'full_name': '13. Wheels and Tires',
        'base_cost': 330,
        'degradation_rate': 0.15,
        'failure_threshold': 0.70
    },
    'Miscellaneous': {
        'name': '14. Miscellaneous',
        'full_name': '14. Miscellaneous',
        'base_cost': 391,
        'degradation_rate': 0.08,
        'failure_threshold': 0.75
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
