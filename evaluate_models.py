import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_pipeline import load_data, add_features, build_lag_features, split_data, TimeSeriesScaler
from model_sarima import train_sarima, save_sarima_model, load_sarima_model, forecast_sarima
from model_random_forest import RainfallRFModel, recursive_forecast_rf
from model_lstm import RainfallLSTMModel, prepare_lstm_data, train_lstm, recursive_forecast_lstm
from crop_planning import calculate_onset_cessation_for_year, dekad_to_date_str

def main():
    print("=========================================")
    # 1. Load and process data
    print("1. Loading and preprocessing data...")
    df = load_data()
    df = add_features(df)
    df_lags, lag_cols = build_lag_features(df, max_lag=36)
    
    train, val, test = split_data(df_lags)
    print(f"Data split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # 2. Train and Save SARIMA model
    # We train on Train + Val to give SARIMA the maximum history up to the test set start
    print("\n2. Training SARIMAX model on historical data...")
    combined_train_val = pd.concat([train, val], ignore_index=True)
    sarima_results = train_sarima(combined_train_val['rfh'])
    save_sarima_model(sarima_results, "sarima_model.pkl")
    
    # 3. Train and Save Random Forest
    print("\n3. Training Random Forest Regressor...")
    feature_cols = lag_cols + ['rfh_avg', 'dekad_sin', 'dekad_cos']
    rf_model = RainfallRFModel()
    rf_model.fit(combined_train_val, combined_train_val['rfh'], feature_cols)
    rf_model.save("random_forest.pkl")
    
    # 4. Train and Save LSTM (Uses train-validation split for checkpointing)
    print("\n4. Training LSTM neural network in PyTorch...")
    scaler = TimeSeriesScaler()
    train_seq, train_y = prepare_lstm_data(train, lag_cols, scaler, is_train=True)
    val_seq, val_y = prepare_lstm_data(val, lag_cols, scaler, is_train=False)
    
    # Save the LSTM scaler
    with open("lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    lstm_model = train_lstm(
        train_seq, train_y, 
        val_seq, val_y, 
        epochs=40, 
        batch_size=32, 
        lr=0.005, 
        model_path="lstm_model.pth"
    )
    
    # 5. Generate forecasts on the blind test set (2024-01-01 to 2026-07-01, length = 91)
    print("\n5. Generating test set forecasts...")
    steps = len(test)
    
    # A. SARIMA Forecast
    sarima_pred = forecast_sarima(sarima_results, steps=steps).values
    sarima_pred = np.maximum(0, sarima_pred) # No negative rainfall
    
    # B. Random Forest Recursive Forecast
    # Get last 36 actual values from validation set
    last_val_idx = len(val) - 1
    # rfh at last row, then lag_1, ..., lag_35
    start_lags = [val.loc[last_val_idx, 'rfh']] + [val.loc[last_val_idx, f'rfh_lag_{j}'] for j in range(1, 36)]
    
    rf_pred = recursive_forecast_rf(rf_model, start_lags, steps=steps, future_exog=test)
    
    # C. LSTM Recursive Forecast
    lstm_pred = recursive_forecast_lstm(lstm_model, start_lags, steps=steps, future_exog=test, scaler=scaler)
    
    # 6. Evaluation metrics
    y_true = test['rfh'].values
    
    metrics = []
    for name, pred in [("SARIMAX", sarima_pred), ("Random Forest", rf_pred), ("LSTM", lstm_pred)]:
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)
        metrics.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })
        
    metrics_df = pd.DataFrame(metrics)
    print("\n=== MODEL PERFORMANCE METRICS ON TEST SET (2024-2026) ===")
    print(metrics_df.to_string(index=False))
    
    # Save metrics to CSV
    metrics_df.to_csv("model_evaluation_metrics.csv", index=False)
    
    # 7. Crop planning validation for test years (2024 and 2025)
    print("\n6. Validating agricultural calendars for test years...")
    
    test_clean = test.copy()
    test_clean['sarima_pred'] = sarima_pred
    test_clean['rf_pred'] = rf_pred
    test_clean['lstm_pred'] = lstm_pred
    
    # Calculate crop calendars
    calendar_validation = []
    for test_year in [2024, 2025]:
        year_true = test_clean[test_clean['year'] == test_year]
        
        # We need to make sure we have 36 dekads for calculations.
        # Since test set starts from 2024, it has complete data for 2024 and 2025
        if len(year_true) == 36:
            # True values
            true_onset, true_cess = calculate_onset_cessation_for_year(year_true)
            
            # Predict values for each model
            # SARIMA
            df_sarima = year_true.copy()
            df_sarima['rfh'] = df_sarima['sarima_pred']
            sarima_onset, sarima_cess = calculate_onset_cessation_for_year(df_sarima)
            
            # RF
            df_rf = year_true.copy()
            df_rf['rfh'] = df_rf['rf_pred']
            rf_onset, rf_cess = calculate_onset_cessation_for_year(df_rf)
            
            # LSTM
            df_lstm = year_true.copy()
            df_lstm['rfh'] = df_lstm['lstm_pred']
            lstm_onset, lstm_cess = calculate_onset_cessation_for_year(df_lstm)
            
            calendar_validation.append({
                "Year": test_year,
                "True Onset": dekad_to_date_str(true_onset),
                "SARIMA Onset": dekad_to_date_str(sarima_onset),
                "RF Onset": dekad_to_date_str(rf_onset),
                "LSTM Onset": dekad_to_date_str(lstm_onset),
                "True Cessation": dekad_to_date_str(true_cess),
                "SARIMA Cessation": dekad_to_date_str(sarima_cess),
                "RF Cessation": dekad_to_date_str(rf_cess),
                "LSTM Cessation": dekad_to_date_str(lstm_cess),
            })
            
    val_cal_df = pd.DataFrame(calendar_validation)
    print("\n=== ONSET & CESSATION FORECAST VALIDATION ===")
    print(val_cal_df.to_string(index=False))
    val_cal_df.to_csv("crop_calendar_validation.csv", index=False)
    
    # 8. Save Forecast Visualizations
    plt.figure(figsize=(14, 6))
    plt.plot(test['date'], y_true, label='Actual Rainfall', color='black', linewidth=2)
    plt.plot(test['date'], sarima_pred, label='SARIMAX Forecast', linestyle='--', alpha=0.8)
    plt.plot(test['date'], rf_pred, label='Random Forest Forecast', linestyle='-.', alpha=0.8)
    plt.plot(test['date'], lstm_pred, label='LSTM Forecast', linestyle=':', alpha=0.8)
    plt.title('Kogi State Rainfall Time Series Forecast Comparison (2024 - 2026 Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Precipitation (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_filepath = "rainfall_forecast_comparison.png"
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved forecast comparison plot to {os.path.abspath(plot_filepath)}")
    print("=========================================")

if __name__ == "__main__":
    main()
