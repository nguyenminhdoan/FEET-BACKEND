"""
Standalone training script for cost prediction models
No external dependencies - all code included
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

from config import MODEL_DIR, SUBSYSTEMS


def load_training_data(csv_path='data/Bus_Maintenance_TimeSeries_2023-2025.csv'):
    """Load training data from CSV"""
    if not os.path.exists(csv_path):
        # Try alternative path
        csv_path = 'Bus_Maintenance_TimeSeries_2023-2025.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training data not found at {csv_path}")

    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} records")
    return df


def create_sequences(data, sequence_length=6):
    """Create sequences for LSTM training"""
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :-1])  # All features except target
        y.append(data[i+sequence_length, -1])      # Target (total cost)

    return np.array(X), np.array(y)


def build_lstm_model(sequence_length, n_features):
    """Build LSTM model architecture"""
    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(
            64,
            return_sequences=True,
            input_shape=(sequence_length, n_features)
        ),
        layers.Dropout(0.2),

        # Second LSTM layer
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),

        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),

        # Output layer
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def train_subsystem_model(subsystem_name, df, sequence_length=6):
    """Train model for a specific subsystem"""
    print(f"\n{'='*60}")
    print(f"Training model for: {subsystem_name}")
    print(f"{'='*60}")

    # Filter data for this subsystem
    subsystem_df = df[df['Subsystem'] == subsystem_name].copy()
    subsystem_df = subsystem_df.sort_values('Date').reset_index(drop=True)

    if len(subsystem_df) == 0:
        print(f"No data found for {subsystem_name}")
        return None

    print(f"Records for {subsystem_name}: {len(subsystem_df)}")

    # Add time features
    subsystem_df['month'] = subsystem_df['Date'].dt.month
    subsystem_df['days_since_start'] = (subsystem_df['Date'] - subsystem_df['Date'].min()).dt.days

    # Select features for training
    feature_cols = [
        'PM_Labour_Cost', 'PM_Parts_Cost',
        'CM_Labour_Cost', 'CM_Parts_Cost',
        'Age_months', 'Mileage_km',
        'month', 'days_since_start'
    ]

    # Calculate target (total cost)
    subsystem_df['Total_Cost'] = (
        subsystem_df['PM_Labour_Cost'] +
        subsystem_df['PM_Parts_Cost'] +
        subsystem_df['CM_Labour_Cost'] +
        subsystem_df['CM_Parts_Cost']
    )

    # Prepare data array
    all_cols = feature_cols + ['Total_Cost']
    data = subsystem_df[all_cols].values

    # Create sequences
    print(f"Creating sequences (length={sequence_length})...")
    X, y = create_sequences(data, sequence_length)

    if len(X) == 0:
        print(f"Not enough data to create sequences for {subsystem_name}")
        return None

    print(f"Created {len(X)} training sequences")

    # Split train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Scale data
    n_features = X_train.shape[2]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    # Fit and transform
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_test_scaled = scaler_X.transform(X_test_reshaped)

    # Reshape back
    X_train_scaled = X_train_scaled.reshape(-1, sequence_length, n_features)
    X_test_scaled = X_test_scaled.reshape(-1, sequence_length, n_features)

    # Scale targets
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    # Build model
    print(f"Building LSTM model...")
    model = build_lstm_model(sequence_length, n_features)
    print(f"Model parameters: {model.count_params():,}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]

    # Train
    print(f"Training for 30 epochs...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=30,
        batch_size=8,
        callbacks=callbacks,
        verbose=0  # Suppress detailed output
    )

    # Evaluate
    train_loss, train_mae = model.evaluate(X_train_scaled, y_train_scaled, verbose=0)
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)

    # Convert MAE back to original scale
    train_mae_orig = train_mae * scaler_y.scale_[0]
    test_mae_orig = test_mae * scaler_y.scale_[0]

    print(f"\nTraining Results:")
    print(f"  Train MAE: ${train_mae_orig:.2f}")
    print(f"  Test MAE: ${test_mae_orig:.2f}")
    print(f"  Epochs trained: {len(history.history['loss'])}")

    # Save model
    model_dir = os.path.join(MODEL_DIR, f'{subsystem_name}_cost_model')
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model.save(os.path.join(model_dir, 'model.h5'))

    # Save scalers
    joblib.dump(scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))

    # Save metadata
    metadata = {
        'subsystem': subsystem_name,
        'sequence_length': sequence_length,
        'n_features': n_features,
        'feature_cols': feature_cols,
        'train_mae': float(train_mae_orig),
        'test_mae': float(test_mae_orig),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model saved to: {model_dir}")

    return model


def main():
    """Main training function"""
    print("="*60)
    print("Fleet Maintenance Cost Prediction - Model Training")
    print("="*60)
    print()

    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load training data
    try:
        df = load_training_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure the training data CSV file is in the data/ directory:")
        print("  data/Bus_Maintenance_TimeSeries_2023-2025.csv")
        return

    print()

    # Train models for each subsystem
    trained_count = 0
    for subsystem_name in SUBSYSTEMS.keys():
        try:
            model = train_subsystem_model(subsystem_name, df)
            if model is not None:
                trained_count += 1
        except Exception as e:
            print(f"\nError training {subsystem_name}: {e}")
            continue

    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Successfully trained {trained_count}/{len(SUBSYSTEMS)} models")
    print(f"Models saved to: {MODEL_DIR}")
    print()
    print("You can now start the API server to use the models.")
    print("="*60)


if __name__ == '__main__':
    main()
