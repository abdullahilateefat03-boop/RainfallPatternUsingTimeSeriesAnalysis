import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle

class RainfallDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RainfallLSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take output of last sequence step
        out_last = out[:, -1, :]
        pred = self.linear(out_last)
        return pred

def prepare_lstm_data(df, lag_cols, scaler=None, is_train=False):
    """
    Transforms tabular lag features into a 3D sequence array for PyTorch LSTM.
    Sequence length = 36.
    Each step in the sequence has 4 features: [scaled_rainfall, scaled_avg, sin, cos].
    """
    # Exogenous values for target (needed for scale-aligning)
    X_lags = df[lag_cols].values # shape: (N, 36)
    
    # Let's match average values for each of those 36 lags.
    # To keep it simple and robust, we can just feed a 3D sequence of shape (N, 36, 4):
    # For each lag step j from 36 down to 1:
    # Feature 1: rfh_lag_j
    # Feature 2: rfh_avg (value for that lag step) - we can approximate or use current rfh_avg
    # Feature 3: sin for that lag step
    # Feature 4: cos for that lag step
    # Let's construct this sequence array:
    N = len(df)
    X_seq = np.zeros((N, 36, 4))
    
    # We can fit a scaler on rainfall if it's training
    if is_train and scaler is not None:
        scaler.fit(df['rfh'].values.reshape(-1, 1))
        
    for i in range(N):
        for j in range(36):
            lag_num = 36 - j # 36, 35, ..., 1
            lag_val = df.loc[i, f'rfh_lag_{lag_num}']
            if scaler is not None:
                lag_val_scaled = scaler.transform(np.array([[lag_val]]))[0, 0]
            else:
                lag_val_scaled = lag_val
                
            # Exogenous values for that specific period: we can use cyclical temporal indicators
            # For simplicity, we assign the current step's indicators or shift them. 
            # Using current step's indicators is a very standard simplification.
            X_seq[i, j, 0] = lag_val_scaled
            X_seq[i, j, 1] = df.loc[i, 'rfh_avg'] / 100.0 # simple scaling for averages
            X_seq[i, j, 2] = df.loc[i, 'dekad_sin']
            X_seq[i, j, 3] = df.loc[i, 'dekad_cos']
            
    y = df['rfh'].values
    if scaler is not None:
        y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    else:
        y_scaled = y
        
    return X_seq, y_scaled

def train_lstm(train_seq, train_y, val_seq, val_y, epochs=50, batch_size=32, lr=0.001, model_path="lstm_model.pth"):
    train_dataset = RainfallDataset(train_seq, train_y)
    val_dataset = RainfallDataset(val_seq, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = RainfallLSTMModel(input_size=4, hidden_size=32, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        # Checkpoint: Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Best Val Loss: {best_val_loss:.4f})")
            
    print(f"LSTM training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Load best weights back
    model.load_state_dict(torch.load(model_path))
    return model

def recursive_forecast_lstm(model, start_lags, steps, future_exog, scaler):
    """
    Recursive multi-step forecast using PyTorch LSTM.
    - start_lags: List of the 36 most recent actual lags (newest first)
    - steps: Number of steps to forecast
    - future_exog: DataFrame containing future exogenous variables
    - scaler: Fitted TimeSeriesScaler for scaling/unscaling rainfall values
    """
    model.eval()
    forecasted_values = []
    current_lags = list(start_lags) # Keep a working copy of size 36
    
    for i in range(steps):
        # Create a single sequence of shape (1, 36, 4)
        X_seq = np.zeros((1, 36, 4))
        exog_row = future_exog.iloc[i]
        
        for j in range(36):
            lag_num = 36 - j
            lag_val = current_lags[lag_num - 1]
            lag_val_scaled = scaler.transform(np.array([[lag_val]]))[0, 0]
            
            X_seq[0, j, 0] = lag_val_scaled
            X_seq[0, j, 1] = exog_row['rfh_avg'] / 100.0
            X_seq[0, j, 2] = exog_row['dekad_sin']
            X_seq[0, j, 3] = exog_row['dekad_cos']
            
        # Predict
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(X_tensor).item()
            
        # Unscale prediction
        pred_val = scaler.inverse_transform(np.array([[pred_scaled]]))[0, 0]
        pred_val = max(0.0, pred_val) # Prevent negative values
        
        forecasted_values.append(pred_val)
        
        # Shift lags
        current_lags = [pred_val] + current_lags[:-1]
        
    return np.array(forecasted_values)

if __name__ == "__main__":
    from data_pipeline import load_data, add_features, build_lag_features, split_data, TimeSeriesScaler
    print("Testing LSTM model pipeline...")
    
    df = load_data()
    df = add_features(df)
    df_lags, lag_cols = build_lag_features(df, max_lag=36)
    
    train, val, test = split_data(df_lags)
    
    scaler = TimeSeriesScaler()
    
    train_seq, train_y = prepare_lstm_data(train, lag_cols, scaler, is_train=True)
    val_seq, val_y = prepare_lstm_data(val, lag_cols, scaler, is_train=False)
    
    # Save the scaler so it can be reloaded for Streamlit app
    with open("lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"Train sequences shape: {train_seq.shape}")
    print(f"Val sequences shape: {val_seq.shape}")
    
    model = train_lstm(train_seq, train_y, val_seq, val_y, epochs=30, lr=0.005)
    
    # Test recursive forecast
    last_train_idx = len(train) - 1
    start_lags = [train.loc[last_train_idx, f'rfh_lag_{j}'] for j in range(1, 37)]
    
    forecast = recursive_forecast_lstm(model, start_lags, steps=len(val), future_exog=val, scaler=scaler)
    print(f"LSTM forecast complete. Shape: {forecast.shape}")
    print(f"Min forecast: {forecast.min():.2f}, Max forecast: {forecast.max():.2f}")
