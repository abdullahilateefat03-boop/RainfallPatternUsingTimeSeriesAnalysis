import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import os

def train_sarima(train_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 36)):
    """
    Fits a SARIMAX model on the training series.
    s=36 is used since there are 36 dekads in a year.
    """
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    print("Fitting SARIMAX model... (this may take a few seconds)")
    results = model.fit(disp=False)
    print("SARIMAX fitting completed.")
    return results

def save_sarima_model(results, filepath="sarima_model.pkl"):
    results.save(filepath)
    print(f"Saved SARIMAX model to {filepath}")

def load_sarima_model(filepath="sarima_model.pkl"):
    if os.path.exists(filepath):
        return SARIMAXResults.load(filepath)
    raise FileNotFoundError(f"Model file not found at {filepath}")

def forecast_sarima(results, steps):
    """
    Forecasts future values.
    """
    return results.forecast(steps=steps)

if __name__ == "__main__":
    from data_pipeline import load_data, split_data
    print("Testing SARIMAX model pipeline...")
    df = load_data()
    # For SARIMAX, we fit directly on the raw rainfall time series rfh
    train, val, test = split_data(df)
    
    # Train on train set rfh
    results = train_sarima(train['rfh'])
    save_sarima_model(results, "sarima_model.pkl")
    
    # Reload and test forecast
    loaded_results = load_sarima_model("sarima_model.pkl")
    predictions = forecast_sarima(loaded_results, steps=len(val))
    print(f"Forecasted {len(predictions)} steps. Min value: {predictions.min():.2f}, Max value: {predictions.max():.2f}")
