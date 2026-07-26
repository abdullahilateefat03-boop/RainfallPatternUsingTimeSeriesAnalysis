import pandas as pd
import numpy as np

def load_data(filepath="kogi_chirps_monthly.csv"):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def add_features(df):
    # Add calendar-based features
    df['month'] = df['date'].dt.month
    
    # Calculate dekad of the year (1 to 36)
    # Each month has 3 dekads: days 1-10 (dekad 1), 11-20 (dekad 2), 21-end (dekad 3)
    df['dekad_in_month'] = df['date'].dt.day.map(lambda d: 1 if d <= 10 else (2 if d <= 20 else 3))
    df['dekad_of_year'] = (df['month'] - 1) * 3 + df['dekad_in_month']
    
    # Sine/Cosine cyclical encoding of the dekad of the year
    df['dekad_sin'] = np.sin(2 * np.pi * df['dekad_of_year'] / 36.0)
    df['dekad_cos'] = np.cos(2 * np.pi * df['dekad_of_year'] / 36.0)
    
    return df

def build_lag_features(df, max_lag=36):
    df_lags = df.copy()
    lag_cols = []
    
    for lag in range(1, max_lag + 1):
        col_name = f'rfh_lag_{lag}'
        df_lags[col_name] = df_lags['rfh'].shift(lag)
        lag_cols.append(col_name)
        
    # Drop rows that have NaNs due to shifting
    df_lags = df_lags.dropna(subset=lag_cols).reset_index(drop=True)
    return df_lags, lag_cols

def split_data(df, train_end_year=2020, val_end_year=2023):
    df['year'] = df['date'].dt.year
    
    train_df = df[df['year'] <= train_end_year].copy().reset_index(drop=True)
    val_df = df[(df['year'] > train_end_year) & (df['year'] <= val_end_year)].copy().reset_index(drop=True)
    test_df = df[df['year'] > val_end_year].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

class TimeSeriesScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1.0, self.std)
        
    def transform(self, x):
        return (x - self.mean) / self.std
        
    def inverse_transform(self, x_scaled):
        return (x_scaled * self.std) + self.mean

if __name__ == "__main__":
    # Small test runner to verify everything runs
    print("Testing data pipeline...")
    df = load_data()
    df = add_features(df)
    print(f"Loaded dataset with features. Shape: {df.shape}")
    
    df_lags, lag_cols = build_lag_features(df, max_lag=36)
    print(f"Engineered {len(lag_cols)} lags. Shape after dropping: {df_lags.shape}")
    
    train, val, test = split_data(df_lags)
    print(f"Splits - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
