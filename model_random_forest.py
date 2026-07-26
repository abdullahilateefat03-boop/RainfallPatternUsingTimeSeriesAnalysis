import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

class RainfallRFModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.feature_cols = None
        
    def fit(self, X, y, feature_cols):
        self.feature_cols = feature_cols
        self.model.fit(X[self.feature_cols], y)
        print("Random Forest Regressor training completed.")
        
    def predict(self, X):
        return self.model.predict(X[self.feature_cols])
        
    def save(self, filepath="random_forest.pkl"):
        # Save both the model and the feature columns list
        state = {
            'model': self.model,
            'feature_cols': self.feature_cols
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved Random Forest model to {filepath}")
        
    @classmethod
    def load(cls, filepath="random_forest.pkl"):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            rf_model = cls()
            rf_model.model = state['model']
            rf_model.feature_cols = state['feature_cols']
            return rf_model
        raise FileNotFoundError(f"Model file not found at {filepath}")

def recursive_forecast_rf(rf_model, start_lags, steps, future_exog):
    """
    Performs recursive multi-step forecasting for Random Forest.
    - start_lags: List of the 36 most recent actual lags (newest first, i.e., rfh_lag_1, ..., rfh_lag_36)
    - steps: Number of periods to forecast
    - future_exog: DataFrame containing future exogenous variables ('rfh_avg', 'dekad_sin', 'dekad_cos') for the steps
    """
    forecasted_values = []
    current_lags = list(start_lags)  # Keep a working copy of size 36
    
    # Verify the feature columns list matches what was trained
    feature_cols = rf_model.feature_cols
    
    for i in range(steps):
        # Build the predictor row for step i
        exog_row = future_exog.iloc[i]
        
        # Construct lag dictionary matching feature names: rfh_lag_1, ..., rfh_lag_36
        pred_dict = {}
        for lag_idx in range(1, 37):
            pred_dict[f'rfh_lag_{lag_idx}'] = current_lags[lag_idx - 1]
            
        # Add exogenous variables
        pred_dict['rfh_avg'] = exog_row['rfh_avg']
        pred_dict['dekad_sin'] = exog_row['dekad_sin']
        pred_dict['dekad_cos'] = exog_row['dekad_cos']
        
        # Convert to DataFrame
        X_pred = pd.DataFrame([pred_dict])
        
        # Predict the next value
        pred_val = rf_model.predict(X_pred)[0]
        # Prevent negative rainfall forecasts
        pred_val = max(0.0, pred_val)
        
        forecasted_values.append(pred_val)
        
        # Shift the current lags: pred_val becomes the new rfh_lag_1, and old ones shift down
        current_lags = [pred_val] + current_lags[:-1]
        
    return np.array(forecasted_values)

if __name__ == "__main__":
    from data_pipeline import load_data, add_features, build_lag_features, split_data
    print("Testing Random Forest pipeline...")
    
    df = load_data()
    df = add_features(df)
    df_lags, lag_cols = build_lag_features(df, max_lag=36)
    
    # Feature columns includes both lags and exogenous variables
    feature_cols = lag_cols + ['rfh_avg', 'dekad_sin', 'dekad_cos']
    
    train, val, test = split_data(df_lags)
    
    rf = RainfallRFModel()
    rf.fit(train, train['rfh'], feature_cols)
    rf.save("random_forest.pkl")
    
    # Test loading and recursive forecasting on validation set
    loaded_rf = RainfallRFModel.load("random_forest.pkl")
    
    # Find the starting lags: the last 36 actual values of the training set (sorted from newest to oldest)
    # The training set rows are ordered. Let's look at the very last row of train:
    # train['rfh'] value at the last row, train['rfh_lag_1'] which is the second last row's rfh, etc.
    # A robust way is to just take the actual target values of the last 36 rows of the train set, reversed.
    last_train_idx = len(train) - 1
    start_lags = [train.loc[last_train_idx, f'rfh_lag_{j}'] for j in range(1, 37)]
    
    # Forecast on validation period
    forecast = recursive_forecast_rf(loaded_rf, start_lags, steps=len(val), future_exog=val)
    print(f"Recursive forecast complete. Shapes - Forecast: {forecast.shape}, Val: {val.shape}")
    print(f"Min forecast: {forecast.min():.2f}, Max forecast: {forecast.max():.2f}")
