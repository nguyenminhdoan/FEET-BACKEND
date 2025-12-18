from __future__ import annotations
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_DIR


REQUIRED_PM_CM_COLS = ['Job Reason Description', 'Labor Cost', 'Part Cost', 'Other Cost']
SUBSYSTEM_COL = 'Subsystem'
PREFERRED_JOIN_KEYS = [
    ['Unit Number', 'Job Code', 'Job Open Date'],
    ['Unit Number', 'Job Open Date'],
    ['Unit Number', 'Job Code'],
    ['Job Code', 'Job Open Date'],
    ['Unit Number'],
    ['Job Code'],
    ['Truncated Job Code'],
]
MAPPING_SHEET_NAME = 'Proposed Mapping'
MAPPING_SOURCE_COL = 'Component/Subsystem in Raw Data'
# Try both column name variations (note the typo in Excel: "Subsytem" vs "Subsystem")
# Prefer 'Updated Subsytem Mapping' for detailed subsystem categories (94 subsystems)
# 'System' column only has 14 high-level categories
MAPPING_TARGET_COLS = ['Updated Subsytem Mapping', 'Updated Subsystem Mapping', 'Proposed Subsystem Mapping', 'Subsystem', 'System']


def _normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text for consistent mapping (strip + lowercase)."""
    return series.astype(str).str.strip().str.lower()


def _normalize_key_columns(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Normalize key columns for joining (e.g., date strings to date)."""
    df = df.copy()
    for k in keys:
        if 'date' in k.lower() and k in df.columns:
            df[k] = pd.to_datetime(df[k], errors='coerce').dt.date
    return df


def load_first_matching_sheet(xlsx_path: str, required_cols: list[str]) -> tuple[pd.DataFrame, str]:
    """Load the first sheet that contains all required columns."""
    xls = pd.ExcelFile(xlsx_path)
    preferred = ['Work Orders']
    ordered_sheets = preferred + [s for s in xls.sheet_names if s not in preferred]
    for sheet in ordered_sheets:
        df_candidate = xls.parse(sheet)
        if all(col in df_candidate.columns for col in required_cols):
            return df_candidate, sheet
    raise ValueError(
        f"No sheet in {xlsx_path} contains columns: {required_cols}. "
        f"Sheets found: {xls.sheet_names}"
    )


def maybe_merge_subsystem_from_other_sheet(xlsx_path: str, cost_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list[str] | None]:
    """If Subsystem is missing, look for another sheet that has it and merge on preferred keys."""
    if SUBSYSTEM_COL in cost_df.columns:
        return cost_df, None, None

    # NOTE: Commenting out 'Model Categorization' direct mapping to use detailed keyword-based mapping instead
    # 'Model Categorization' only has 14 high-level categories
    # The 'Updated Subsytem Mapping' from 'Proposed Mapping' sheet has 94 detailed subsystems
    # if 'Model Categorization' in cost_df.columns:
    #     cost_df[SUBSYSTEM_COL] = cost_df['Model Categorization']
    #     print(f"Using 'Model Categorization' column as Subsystem")
    #     return cost_df, 'Work Orders (Model Categorization column)', ['direct mapping']

    xls = pd.ExcelFile(xlsx_path)
    preferred = [MAPPING_SHEET_NAME]
    other_sheets = [s for s in xls.sheet_names if s not in preferred]
    search_order = preferred + other_sheets

    for sheet in search_order:
        df_candidate = xls.parse(sheet)
        # Keyword mapping sheet: match Job Description keywords to Component/Subsystem in Raw Data
        # Find which target column exists in this sheet
        mapping_target_col = None
        if sheet == MAPPING_SHEET_NAME and MAPPING_SOURCE_COL in df_candidate.columns:
            for col in MAPPING_TARGET_COLS:
                if col in df_candidate.columns:
                    mapping_target_col = col
                    break

        if mapping_target_col:
            # Build mapping: normalized keywords -> subsystem
            keywords_list = [((_normalize_text(pd.Series([k]))[0], df_candidate.iloc[i][mapping_target_col]))
                            for i, k in enumerate(df_candidate[MAPPING_SOURCE_COL])]
            
            merged = cost_df.copy()
            merged['_normalized_job_desc'] = _normalize_text(merged.get('Job Description', pd.Series(index=merged.index, dtype=str)))
            
            # Apply keyword matching: for each job description, find first keyword that matches as substring
            def find_subsystem(job_desc):
                if pd.isna(job_desc):
                    return None
                for keyword, subsystem in keywords_list:
                    if keyword in job_desc:
                        return subsystem
                return None
            
            merged[SUBSYSTEM_COL] = merged['_normalized_job_desc'].apply(find_subsystem)
            merged = merged.drop(columns=['_normalized_job_desc'])
            
            # Return immediately if any subsystem was mapped; otherwise keep searching
            if merged[SUBSYSTEM_COL].notna().any():
                return merged, sheet, [MAPPING_SOURCE_COL + ' keywords -> Job Description (substring match)']
            # Fall through to try other sheets if mapping yielded no matches
            cost_df = merged

        if SUBSYSTEM_COL in df_candidate.columns:
            for keys in PREFERRED_JOIN_KEYS:
                if all(k in cost_df.columns for k in keys) and all(k in df_candidate.columns for k in keys):
                    left = _normalize_key_columns(cost_df, keys)
                    right = _normalize_key_columns(df_candidate[[SUBSYSTEM_COL] + keys], keys)
                    merged = pd.merge(left, right, on=keys, how='left')
                    return merged, sheet, keys

    raise ValueError(
        f"Subsystem column not found in the main sheet and no mergeable sheet located in {xlsx_path}. "
        f"Checked sheets: {xls.sheet_names}. Ensure '{SUBSYSTEM_COL}' exists, or place it in the same sheet, "
        f"or add one of the join key sets: {PREFERRED_JOIN_KEYS} so we can merge."
    )


def add_pm_cm_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Create PM/CM cost columns based on Job Reason Description (warranty => CM, else PM).
    
    NOTE: This creates class imbalance if warranty items are rare.
    Consider alternative strategies:
    - Use actual warranty period (if available)
    - Use Job Status or Work Type fields
    - Manually review and tag warranty/non-warranty records
    """
    required_cols = REQUIRED_PM_CM_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for PM/CM mapping: {missing}")

    df = df.copy()

    is_cm = df['Job Reason Description'].str.contains('warranty', case=False, na=False)

    df['PM_Labour_Cost'] = 0.0
    df['PM_Parts_Cost'] = 0.0
    df['CM_Labour_Cost'] = 0.0
    df['CM_Parts_Cost'] = 0.0

    # Treat Other Cost as parts bucket to avoid losing value
    df.loc[~is_cm, 'PM_Labour_Cost'] = df['Labor Cost'].fillna(0)
    df.loc[~is_cm, 'PM_Parts_Cost'] = df['Part Cost'].fillna(0) + df['Other Cost'].fillna(0)

    df.loc[is_cm, 'CM_Labour_Cost'] = df['Labor Cost'].fillna(0)
    df.loc[is_cm, 'CM_Parts_Cost'] = df['Part Cost'].fillna(0) + df['Other Cost'].fillna(0)

    return df


def load_training_data(path: str | None = None):
    """Load training data from Excel or CSV with sensible fallbacks."""
    default_xlsx = 'data/YRT_FZ_sample_data.xlsx'
    default_csv = 'data/YRT_FZ_sample_data.csv'

    candidates = []
    if path:
        candidates.append(path)
    else:
        candidates.extend([default_xlsx, default_csv, os.path.basename(default_xlsx), os.path.basename(default_csv)])

    file_path = next((p for p in candidates if os.path.exists(p)), None)
    if not file_path:
        raise FileNotFoundError(f"Training data not found. Looked for: {candidates}")

    print(f"Loading training data from {file_path}...")

    if file_path.lower().endswith(('.xlsx', '.xls')):
        df, sheet_used = load_first_matching_sheet(file_path, REQUIRED_PM_CM_COLS)
        print(f"Using sheet for costs: {sheet_used}")
        df, subsystem_sheet, merge_keys = maybe_merge_subsystem_from_other_sheet(file_path, df)
        if subsystem_sheet:
            print(f"Subsystem merged from sheet: {subsystem_sheet} on keys: {merge_keys}")
    else:
        df = pd.read_csv(file_path)

    df = add_pm_cm_costs(df)

    # If Date is missing but Job Open Date exists, use it
    if 'Date' not in df.columns and 'Job Open Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Job Open Date'], errors='coerce')
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Merge mileage from Distance Driven sheet if it exists and Mileage column is missing
    if file_path.lower().endswith(('.xlsx', '.xls')) and 'Mileage' not in df.columns:
        try:
            xls = pd.ExcelFile(file_path)
            if 'Distance Driven' in xls.sheet_names:
                dd = xls.parse('Distance Driven')
                # Distance Driven sheet should have Unit Number and mileage by year columns
                # Reshape to long format and merge by Unit Number and year
                if 'Unit Number' in dd.columns:
                    # Get the most recent year's mileage for each bus, or sum/average across years
                    year_cols = [c for c in dd.columns if str(c).isdigit() and int(c) >= 2020]
                    if year_cols:
                        # Use the most recent year available
                        most_recent_year = max(year_cols)
                        dd_mileage = dd[['Unit Number', most_recent_year]].rename(
                            columns={most_recent_year: 'Mileage'}
                        )
                        df = pd.merge(df, dd_mileage, on='Unit Number', how='left')
                        print(f"Merged Mileage from Distance Driven sheet (year {most_recent_year})")
        except Exception as e:
            print(f"Warning: Could not merge Distance Driven sheet: {e}")
    
    # If Mileage still missing, estimate it
    if 'Mileage' not in df.columns:
        print("Warning: Mileage column not found, estimating from Age_months...")
        if 'Age_months' in df.columns:
            df['Mileage'] = df['Age_months'] * 2500  # ~30k km/year
        else:
            df['Mileage'] = np.random.randint(20000, 250000, size=len(df))

    # Rename Mileage to Mileage_km for consistency
    if 'Mileage' in df.columns:
        df['Mileage_km'] = df['Mileage']
    
    # Calculate Age_months from Date if missing
    if 'Age_months' not in df.columns and 'Date' in df.columns:
        min_date = df['Date'].min()
        df['Age_months'] = ((df['Date'] - min_date).dt.days / 30.44).astype(int)
        print("Calculated Age_months from Date column")

    # Guardrail: ensure we have Subsystem values after mapping
    if SUBSYSTEM_COL not in df.columns or df[SUBSYSTEM_COL].dropna().empty:
        raise ValueError(
            "No subsystems found after applying mapping. Ensure the 'Proposed Mapping' sheet pairs 'Component/Subsystem in Raw Data' to actual Subsystem values and that Job Description strings in the cost sheet match after lower/strip normalization."
        )
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


def create_ml_features(subsystem_df):
    """Create features for ML models (RandomForest, XGBoost)"""
    df = subsystem_df.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # Time-based features
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features (previous costs)
    for lag in [1, 2, 3, 6]:
        df[f'total_cost_lag_{lag}'] = df['Total_Cost'].shift(lag)
        df[f'pm_cost_lag_{lag}'] = (df['PM_Labour_Cost'] + df['PM_Parts_Cost']).shift(lag)
        df[f'cm_cost_lag_{lag}'] = (df['CM_Labour_Cost'] + df['CM_Parts_Cost']).shift(lag)

    # Rolling statistics
    for window in [3, 6]:
        df[f'total_cost_rolling_mean_{window}'] = df['Total_Cost'].rolling(window=window, min_periods=1).mean()
        df[f'total_cost_rolling_std_{window}'] = df['Total_Cost'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'total_cost_rolling_max_{window}'] = df['Total_Cost'].rolling(window=window, min_periods=1).max()

    # Mileage rate of change
    df['mileage_change'] = df['Mileage_km'].diff().fillna(0)

    # Age groups
    df['age_group'] = pd.cut(df['Age_months'], bins=[0, 24, 48, 72, 96, 200],
                              labels=['0-2yr', '2-4yr', '4-6yr', '6-8yr', '8+yr'])
    df['age_group'] = df['age_group'].cat.codes

    # Binary: Had maintenance cost last month
    df['had_cost_last_month'] = (df['Total_Cost'].shift(1) > 0).astype(int)

    # Fill NaN values
    df = df.fillna(0)

    return df


def get_ml_feature_cols():
    """Get list of features for ML models"""
    return [
        # Current period costs
        'PM_Labour_Cost', 'PM_Parts_Cost', 'CM_Labour_Cost', 'CM_Parts_Cost',
        # Vehicle state
        'Age_months', 'Mileage_km', 'mileage_change', 'age_group',
        # Temporal features
        'month', 'quarter', 'year', 'day_of_year', 'days_since_start',
        'month_sin', 'month_cos',
        # Lag features
        'total_cost_lag_1', 'total_cost_lag_2', 'total_cost_lag_3', 'total_cost_lag_6',
        'pm_cost_lag_1', 'pm_cost_lag_2', 'pm_cost_lag_3', 'pm_cost_lag_6',
        'cm_cost_lag_1', 'cm_cost_lag_2', 'cm_cost_lag_3', 'cm_cost_lag_6',
        # Rolling statistics
        'total_cost_rolling_mean_3', 'total_cost_rolling_std_3', 'total_cost_rolling_max_3',
        'total_cost_rolling_mean_6', 'total_cost_rolling_std_6', 'total_cost_rolling_max_6',
        # Binary features
        'had_cost_last_month'
    ]


def train_ml_models(subsystem_name, subsystem_df, min_samples=1):
    """Train ML models (RandomForest and XGBoost) for comparison with LSTM"""
    print(f"\n  🤖 Training ML Models (RandomForest & XGBoost)...")
    
    # Calculate Total_Cost
    subsystem_df['Total_Cost'] = (
        subsystem_df['PM_Labour_Cost'] + subsystem_df['PM_Parts_Cost'] +
        subsystem_df['CM_Labour_Cost'] + subsystem_df['CM_Parts_Cost']
    )

    # Create features
    df_features = create_ml_features(subsystem_df)
    feature_cols = get_ml_feature_cols()

    # Prepare data
    X = df_features[feature_cols].values
    y = df_features['Total_Cost'].values

    # Remove NaN/inf
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
    X = X[mask]
    y = y[mask]

    if len(X) < min_samples:
        print(f"     ⚠️  Only {len(X)} valid ML samples (minimum {min_samples})")
        return None

    # For very small datasets, use leave-one-out cross-validation concept
    # But still use 80/20 split for consistency
    n_samples = len(X)
    
    if n_samples < 10:
        # Extreme regularization for tiny datasets
        rf_max_depth = 2
        xgb_max_depth = 1
        xgb_learning_rate = 0.01
        print(f"     🚨 TINY dataset ({n_samples} samples) - extreme regularization applied")
    elif n_samples < 30:
        # Strong regularization for small data
        rf_max_depth = 5
        xgb_max_depth = 3
        xgb_learning_rate = 0.05
        print(f"     ⚠️  Small dataset ({n_samples} samples) - strong regularization applied")
    elif n_samples < 50:
        # Moderate-strong regularization
        rf_max_depth = 6
        xgb_max_depth = 4
        xgb_learning_rate = 0.07
        print(f"     📊 Small-medium dataset ({n_samples} samples) - moderate regularization")
    elif n_samples < 100:
        rf_max_depth = 8
        xgb_max_depth = 5
        xgb_learning_rate = 0.08
        print(f"     📊 Medium dataset ({n_samples} samples) - moderate regularization")
    else:
        rf_max_depth = 10
        xgb_max_depth = 6
        xgb_learning_rate = 0.1
        print(f"     ✓ Large dataset ({n_samples} samples) - standard parameters")

    # For very small data (< 10), use stratified approach
    if n_samples < 10:
        # Use 70/30 split for tiny data (more training data)
        test_size = max(1, int(n_samples * 0.3))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest with adaptive parameters
    rf_model = RandomForestRegressor(
        n_estimators=50 if n_samples < 50 else 100,  # Fewer trees for small data
        max_depth=rf_max_depth,
        min_samples_split=max(2, len(X_train)//5),
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    results = {
        'rf_mae': rf_mae,
        'rf_r2': rf_r2,
        'rf_model': rf_model,
        'n_samples': n_samples
    }

    # Train XGBoost with adaptive parameters
    if XGBOOST_AVAILABLE:
        xgb_model = XGBRegressor(
            n_estimators=50 if n_samples < 50 else 100,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.5 if n_samples < 10 else (0.1 if n_samples < 50 else 0.01),
            reg_lambda=2.0 if n_samples < 10 else (1.0 if n_samples < 50 else 0.5)
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)

        results['xgb_mae'] = xgb_mae
        results['xgb_r2'] = xgb_r2
        results['xgb_model'] = xgb_model

        print(f"     RandomForest: MAE=${rf_mae:.2f}, R²={rf_r2:.4f}")
        print(f"     XGBoost:      MAE=${xgb_mae:.2f}, R²={xgb_r2:.4f}")
    else:
        print(f"     RandomForest: MAE=${rf_mae:.2f}, R²={rf_r2:.4f}")

    results['scaler'] = scaler
    results['feature_cols'] = feature_cols
    results['n_train'] = len(X_train)
    results['n_test'] = len(X_test)

    return results


def train_subsystem_model(subsystem_name, df, sequence_length=6, train_ml=True, min_samples=1):
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

    # Skip LSTM for very small data (< sequence_length records)
    # LSTM needs sequences, so minimum is sequence_length records
    train_lstm = len(subsystem_df) >= sequence_length
    
    if not train_lstm and len(subsystem_df) > 0:
        print(f"  ⚠️  Only {len(subsystem_df)} records (< sequence_length={sequence_length})")
        print(f"  ⏭️  Skipping LSTM - using ML models only for this subsystem")

    # Train ML models first for comparison
    ml_results = None
    if train_ml:
        try:
            ml_results = train_ml_models(subsystem_name, subsystem_df.copy(), min_samples=min_samples)
        except Exception as e:
            print(f"     ⚠️  ML training failed: {e}")

    # Skip LSTM training if not enough data
    if not train_lstm:
        # Save ML-only model
        if ml_results:
            model_dir = os.path.join(MODEL_DIR, f'{subsystem_name}_cost_model')
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(ml_results['rf_model'], os.path.join(model_dir, 'rf_model.pkl'))
            joblib.dump(ml_results['scaler'], os.path.join(model_dir, 'ml_scaler.pkl'))
            if 'xgb_model' in ml_results:
                joblib.dump(ml_results['xgb_model'], os.path.join(model_dir, 'xgb_model.pkl'))
            
            # Save metadata (ML only)
            metadata = {
                'subsystem': subsystem_name,
                'model_type': 'ML_ONLY',
                'original_records': int(len(subsystem_df)),
                'ml_comparison': {
                    'rf_mae': float(ml_results['rf_mae']),
                    'rf_r2': float(ml_results['rf_r2']),
                    'ml_train_samples': int(ml_results['n_train']),
                    'ml_test_samples': int(ml_results['n_test']),
                    'ml_features': len(ml_results['feature_cols'])
                }
            }
            if 'xgb_mae' in ml_results:
                metadata['ml_comparison']['xgb_mae'] = float(ml_results['xgb_mae'])
                metadata['ml_comparison']['xgb_r2'] = float(ml_results['xgb_r2'])
            
            with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ ML-only model saved to: {model_dir}")
        
        return None

    # Continue with LSTM training
    # Add time features
    subsystem_df['month'] = subsystem_df['Date'].dt.month
    subsystem_df['days_since_start'] = (subsystem_df['Date'] - subsystem_df['Date'].min()).dt.days

    # Select features for training (no Make/Model in YRT data)
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

    # Calculate R² score for LSTM
    y_train_pred = model.predict(X_train_scaled, verbose=0)
    y_test_pred = model.predict(X_test_scaled, verbose=0)
    
    # Inverse scale predictions
    y_train_pred_orig = scaler_y.inverse_transform(y_train_pred)
    y_test_pred_orig = scaler_y.inverse_transform(y_test_pred)
    
    # Inverse scale actual values
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    lstm_test_r2 = r2_score(y_test_orig, y_test_pred_orig)

    print(f"\nTraining Results:")
    print(f"  Train MAE: ${train_mae_orig:.2f}")
    print(f"  Test MAE: ${test_mae_orig:.2f}")
    print(f"  Test R²: {lstm_test_r2:.4f}")
    print(f"  Epochs trained: {len(history.history['loss'])}")
    print(f"  Sequence samples: {len(X_train)} train, {len(X_test)} test")
    
    # Calculate average cost per record for context
    avg_cost = subsystem_df['Total_Cost'].mean()
    mae_pct = (train_mae_orig / avg_cost * 100) if avg_cost > 0 else 0
    print(f"  Avg cost per record: ${avg_cost:.2f} (MAE is {mae_pct:.1f}% of avg)")
    
    # Confidence assessment
    if len(X_train) < 30:
        print(f"  ⚠️  WARNING: Only {len(X_train)} sequences — model may be unreliable")
    elif len(X_train) < 50:
        print(f"  ⚠️  Caution: Limited training data ({len(X_train)} sequences)")
    else:
        print(f"  ✓ Adequate data for training")

    # Save model
    model_dir = os.path.join(MODEL_DIR, f'{subsystem_name}_cost_model')
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model.save(os.path.join(model_dir, 'model.h5'))

    # Save scalers
    joblib.dump(scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))

    # Save ML models if trained
    if ml_results:
        joblib.dump(ml_results['rf_model'], os.path.join(model_dir, 'rf_model.pkl'))
        joblib.dump(ml_results['scaler'], os.path.join(model_dir, 'ml_scaler.pkl'))
        if 'xgb_model' in ml_results:
            joblib.dump(ml_results['xgb_model'], os.path.join(model_dir, 'xgb_model.pkl'))

    # Save metadata
    metadata = {
        'subsystem': subsystem_name,
        'model_type': 'LSTM',
        'sequence_length': sequence_length,
        'n_features': n_features,
        'feature_cols': feature_cols,
        'train_mae': float(train_mae_orig),
        'test_mae': float(test_mae_orig),
        'test_r2': float(lstm_test_r2),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'original_records': int(len(subsystem_df)),
        'avg_cost': float(avg_cost),
        'mae_pct_of_avg': float(mae_pct),
        'total_epochs': int(len(history.history['loss']))
    }

    # Add ML comparison results
    if ml_results:
        metadata['ml_comparison'] = {
            'rf_mae': float(ml_results['rf_mae']),
            'rf_r2': float(ml_results['rf_r2']),
            'ml_train_samples': int(ml_results['n_train']),
            'ml_test_samples': int(ml_results['n_test']),
            'ml_features': len(ml_results['feature_cols'])
        }
        if 'xgb_mae' in ml_results:
            metadata['ml_comparison']['xgb_mae'] = float(ml_results['xgb_mae'])
            metadata['ml_comparison']['xgb_r2'] = float(ml_results['xgb_r2'])

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

    # Show PM/CM distribution
    print("Data Distribution:")
    pm_count = (df['PM_Labour_Cost'] + df['PM_Parts_Cost'] > 0).sum()
    cm_count = (df['CM_Labour_Cost'] + df['CM_Parts_Cost'] > 0).sum()
    print(f"  PM records: {pm_count:,} ({pm_count/len(df)*100:.1f}%)")
    print(f"  CM records: {cm_count:,} ({cm_count/len(df)*100:.1f}%)")
    print(f"  Total: {len(df):,} records")
    print()

    # Discover subsystems dynamically from the data
    subsystems = sorted(df['Subsystem'].dropna().unique())

    if not subsystems:
        print("No subsystems found in the training data (column 'Subsystem')")
        return

    print(f"Found {len(subsystems)} subsystems: {', '.join(subsystems)}")
    print()

    # Train models for each discovered subsystem
    trained_count = 0
    results = []
    
    # Configurable minimum samples threshold
    # MIN_SAMPLES = 1 means train on ANY amount of data
    # but with automatic regularization based on size
    MIN_SAMPLES = 1
    
    for subsystem_name in subsystems:
        try:
            model = train_subsystem_model(subsystem_name, df, min_samples=MIN_SAMPLES)
            if model is not None:
                trained_count += 1
            
            # ALSO count ML-only models
            model_dir = os.path.join(MODEL_DIR, f'{subsystem_name}_cost_model')
            metadata_path = os.path.join(model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Handle both LSTM+ML and ML-only models
                if metadata.get('model_type') == 'ML_ONLY':
                    # ML-only model (tiny dataset)
                    ml = metadata.get('ml_comparison', {})
                    best_model = 'XGBoost' if ml.get('xgb_r2', -1) > ml.get('rf_r2', -1) else 'Random Forest'
                    best_mae = ml.get('xgb_mae') if best_model == 'XGBoost' else ml.get('rf_mae')
                    best_r2 = ml.get('xgb_r2') if best_model == 'XGBoost' else ml.get('rf_r2')
                    data_note = "🚨 "  # Critical warning for very small data
                else:
                    # Standard LSTM + ML model
                    best_model = 'LSTM'
                    best_mae = metadata.get('test_mae', 0)
                    best_r2 = metadata.get('test_r2', None)
                    data_note = ""
                    
                    if 'ml_comparison' in metadata:
                        ml = metadata['ml_comparison']
                        rf_mae = ml.get('rf_mae')
                        rf_r2 = ml.get('rf_r2')
                        xgb_mae = ml.get('xgb_mae')
                        xgb_r2 = ml.get('xgb_r2')
                        
                        # Select best based on R² (higher is better)
                        models_r2 = {'LSTM': best_r2}
                        if rf_r2 is not None:
                            models_r2['RF'] = rf_r2
                        if xgb_r2 is not None:
                            models_r2['XGB'] = xgb_r2
                        
                        # Find the model with highest R²
                        best_model_key = max(models_r2, key=models_r2.get)
                        
                        if best_model_key == 'LSTM':
                            best_model = 'LSTM'
                            best_mae = metadata.get('test_mae', 0)
                        elif best_model_key == 'RF':
                            best_model = 'Random Forest'
                            best_mae = rf_mae
                            best_r2 = rf_r2
                        else:
                            best_model = 'XGBoost'
                            best_mae = xgb_mae
                            best_r2 = xgb_r2
                        
                        # Mark if low data
                        n_samples = ml.get('ml_train_samples', 0)
                        if n_samples < 10:
                            data_note = "🚨 "
                        elif n_samples < 30:
                            data_note = "⚠️  "
                
                results.append({
                    'subsystem': subsystem_name,
                    'model': best_model,
                    'mae': best_mae,
                    'r2': best_r2,
                    'data_note': data_note
                })
        except Exception as e:
            print(f"\nError training {subsystem_name}: {e}")
            continue

    print()
    print("="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Successfully trained {len(results)}/{len(subsystems)} models")
    print(f"Models saved to: {MODEL_DIR}")
    
    # Print summary table if we have ML comparison results
    if results:
        print()
        print(f"{'Subsystem':<35} {'Model':<15} {'Test MAE':<12} {'Test R²':<10}")
        print(f"{'-'*35} {'-'*15} {'-'*12} {'-'*10}")
        
        for result in results:
            name = result['subsystem'][:33]
            model = result['model'][:13]
            mae = result['mae']
            r2 = result['r2']
            note = result.get('data_note', '')
            
            mae_str = f"${mae:>9.2f}" if mae else "N/A"
            r2_str = f"{r2:>8.4f}" if r2 is not None else "N/A"
            
            print(f"{note}{name:<34} {model:<15} {mae_str:<12} {r2_str:<10}")
        
        # Calculate averages for models with R²
        models_with_r2 = [r for r in results if r['r2'] is not None]
        if models_with_r2:
            avg_r2 = sum(r['r2'] for r in models_with_r2) / len(models_with_r2)
            avg_mae = sum(r['mae'] for r in models_with_r2) / len(models_with_r2)
            
            print()
            print("="*80)
            print(f"Average Test R²:  {avg_r2:.4f}")
            print(f"Average Test MAE: ${avg_mae:.2f}")
            print()
        tiny_data = [r['subsystem'] for r in results if r.get('data_note', '').strip().startswith('🚨')]
        low_data = [r['subsystem'] for r in results if r.get('data_note', '').strip().startswith('⚠️')]

        print("Data Warnings:")
        print("  🚨 = Tiny data (<10 samples) - predictions VERY unreliable, use for reference only")
        if tiny_data:
            print("    • " + ", ".join(tiny_data))
        print("  ⚠️  = Low data (10-30 samples) - predictions have limited reliability")
        if low_data:
            print("    • " + ", ".join(low_data))
        print("  ✓ = Normal data (30+ samples) - reliable predictions")

if __name__ == '__main__':
    main()
