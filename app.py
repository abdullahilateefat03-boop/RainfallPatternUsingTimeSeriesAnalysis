import streamlit as st
import panda as pd
import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt

# Import project files
import importlib
import data_pipeline
import model_sarima
import model_random_forest
import model_lstm
import crop_planning

importlib.reload(data_pipeline)
importlib.reload(model_sarima)
importlib.reload(model_random_forest)
importlib.reload(model_lstm)
importlib.reload(crop_planning)

from data_pipeline import load_data, add_features, build_lag_features, split_data, TimeSeriesScaler
from model_sarima import load_sarima_model, forecast_sarima
from model_random_forest import RainfallRFModel, recursive_forecast_rf
from model_lstm import RainfallLSTMModel, prepare_lstm_data, recursive_forecast_lstm
from crop_planning import calculate_onset_cessation_for_year, dekad_to_date_str

# Set Streamlit Page Config
st.set_page_config(
    page_title="Kogi State Agrometeorological DSS",
    page_icon="🌾",
    layout="wide"
)

# Custom Styling (Vanilla CSS) for premium dark/vibrant aesthetics
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@300;400;700&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 30px;
            border-radius: 16px;
            color: white;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(56, 239, 125, 0.2);
        }
        
        .main-header h1 {
            margin: 0;
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
        }
        
        .main-header p {
            margin: 8px 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
            font-weight: 300;
        }
        
        .metric-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border-left: 5px solid #11998e;
            margin-bottom: 15px;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 1.8rem;
            color: #111;
            font-weight: 700;
            margin-top: 5px;
        }
        
        .metric-desc {
            font-size: 0.8rem;
            color: #999;
            margin-top: 3px;
        }
        
        .advisory-box {
            background-color: #f0f7f4;
            border: 1px solid #cce3d8;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            color: #1b5e20;
        }
        
        .advisory-box h4 {
            color: #0f3d13;
            margin-top: 0;
        }
        
        .crop-card {
            background: #ffffff;
            border: 1px solid #eaeaea;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }
        .crop-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }
        
        .status-suitable {
            color: #2e7d32;
            font-weight: bold;
        }
        .status-marginal {
            color: #ef6c00;
            font-weight: bold;
        }
        .status-unsuitable {
            color: #c62828;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Application Title Block
st.markdown("""
    <div class="main-header">
        <h1>🌾 Agricultural Decision Support System (ADSS)</h1>
        <p>Explainable Rainfall Forecasting & Local Crop Planting Advisory for Kogi State, Nigeria</p>
    </div>
""", unsafe_allow_html=True)

# Helper functions to load models
@st.cache_resource
def get_sarima():
    return load_sarima_model("sarima_model.pkl")

@st.cache_resource
def get_rf():
    return RainfallRFModel.load("random_forest.pkl")

@st.cache_resource
def get_lstm():
    model = RainfallLSTMModel(input_size=4, hidden_size=32, num_layers=1)
    model.load_state_dict(torch.load("lstm_model.pth"))
    model.eval()
    with open("lstm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# Verify model files exist before launching UI
models_exist = (
    os.path.exists("sarima_model.pkl") and 
    os.path.exists("random_forest.pkl") and 
    os.path.exists("lstm_model.pth") and 
    os.path.exists("lstm_scaler.pkl")
)

if not models_exist:
    st.warning("⚠️ **Trained Models Not Found!**")
    st.info("The application requires model artifacts to run. Please click the button below to train all predictive models (SARIMA, Random Forest, and LSTM) and calculate baseline performances.")
    if st.button("🚀 Train and Save Models"):
        with st.spinner("Training models in background... (this may take ~30 seconds)"):
            import evaluate_models
            evaluate_models.main()
            st.success("Models trained successfully! Reloading application...")
            st.rerun()
else:
    # Sidebar options
    st.sidebar.markdown("### ⚙️ Forecasting Settings")
    
    # Model Selection
    model_choice = st.sidebar.selectbox(
        "Choose Forecast Architecture:",
        ["LSTM Neural Network (PyTorch)", "Random Forest Regressor (ML)", "SARIMAX Model (Statistical)"]
    )
    
    # Forecast Horizon
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (Dekads):",
        min_value=12,
        max_value=36,
        value=36,
        help="Select how many 10-day periods into the future to forecast. 36 dekads corresponds to 1 year."
    )
    
    # Load dataset
    df = load_data()
    df = add_features(df)
    
    # Extract last data point info
    last_date = df['date'].max()
    st.sidebar.markdown(f"**Latest Data Record:** {last_date.strftime('%B %d, %Y')}")
    
    # Information sidebar widget
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Model Explainability")
    if model_choice.startswith("LSTM"):
        st.sidebar.info("🧠 **LSTM (Long Short-Term Memory)** is a recurrent neural network that processes historical temporal patterns recursively. It uses non-linear gates to track long-term seasonal dynamics.")
    elif model_choice.startswith("Random"):
        st.sidebar.info("🌲 **Random Forest** is an ensemble decision-tree regressor. It uses engineered historical lag features (past 36 dekads) and cyclical time markers to map non-linear local interactions.")
    else:
        st.sidebar.info("📈 **SARIMAX** is a classical statistical time-series model that accounts for trend, auto-regressive lags, moving averages, and strict yearly seasonal cycles ($s=36$).")
        
    # Main Forecast Generation Block
    st.subheader(f"🔮 Rainfall Forecast Results ({model_choice})")
    
    with st.spinner("Generating forecasts..."):
        # Setup data splits to get validation lags
        df_lags, lag_cols = build_lag_features(df, max_lag=36)
        train, val, test = split_data(df_lags)
        
        # We start forecasting recursively using the very last actual lags in the dataset
        last_row_idx = len(df_lags) - 1
        start_lags = [df_lags.loc[last_row_idx, 'rfh']] + [df_lags.loc[last_row_idx, f'rfh_lag_{j}'] for j in range(1, 36)]
        
        # Prepare future dates and cyclical features for the forecast horizon
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=10), periods=forecast_horizon, freq='10D')
        future_df = pd.DataFrame({'date': future_dates})
        future_df = add_features(future_df)
        
        # Pull historical averages matching future dekads to serve as climatology baseline
        # (group by dekad of year in original data, calculate mean, and map to future_df)
        dekad_means = df.groupby('dekad_of_year')['rfh'].mean().to_dict()
        future_df['rfh_avg'] = future_df['dekad_of_year'].map(dekad_means)
        
        # Make predictions
        if model_choice.startswith("SARIMAX"):
            sarima_results = get_sarima()
            # Forecast steps ahead
            predictions = forecast_sarima(sarima_results, steps=forecast_horizon).values
            predictions = np.maximum(0, predictions)
        
        elif model_choice.startswith("Random"):
            rf_model = get_rf()
            predictions = recursive_forecast_rf(rf_model, start_lags, steps=forecast_horizon, future_exog=future_df)
            
        else: # LSTM
            lstm_model, scaler = get_lstm()
            predictions = recursive_forecast_lstm(lstm_model, start_lags, steps=forecast_horizon, future_exog=future_df, scaler=scaler)
            
        future_df['rfh_pred'] = predictions
        
        # Calculate Onset and Cessation on Predicted Curve
        # Since crop_planning calculations require a standard calendar year, we align predictions into 
        # a structure starting from April for onset and September for cessation.
        # If the forecast horizon is 36 dekads, it represents a full year.
        onset_dekad, cessation_dekad = calculate_onset_cessation_for_year(
            future_df.rename(columns={'rfh_pred': 'rfh'})
        )
        
        onset_date_str = dekad_to_date_str(onset_dekad, future_dates[0].year)
        cessation_date_str = dekad_to_date_str(cessation_dekad, future_dates[0].year)
        season_length_days = (cessation_dekad - onset_dekad) * 10
        total_predicted_rain = predictions.sum()
        
        # Display Metric Rows
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🌱 Predicted Onset Date</div>
                    <div class="metric-value">{onset_date_str}</div>
                    <div class="metric-desc">Optimal date to begin planting (Walter/Benoit formula)</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">🍂 Predicted Cessation Date</div>
                    <div class="metric-value">{cessation_date_str}</div>
                    <div class="metric-desc">Expected end of crop growing rain window</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">⏳ Season Duration</div>
                    <div class="metric-value">{season_length_days} Days</div>
                    <div class="metric-desc">{cessation_dekad - onset_dekad} active dekads of wet period</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">💧 Cumulative Rainfall</div>
                    <div class="metric-value">{total_predicted_rain:.1f} mm</div>
                    <div class="metric-desc">Total forecasted precipitation depth</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Plotting the forecast
        st.markdown("---")
        st.write("### 📊 Precipitation Profile Visualization")
        
        # Create plot combining recent history (past 24 dekads) and forecast
        hist_tail = df.tail(24)
        
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(hist_tail['date'], hist_tail['rfh'], label='Historical Rainfall (CHIRPS)', color='#555555', linewidth=1.8)
        ax.plot(future_df['date'], future_df['rfh_pred'], label='Forecasted Rainfall', color='#11998e', linewidth=2.5)
        ax.fill_between(future_df['date'], 0, future_df['rfh_pred'], color='#38ef7d', alpha=0.1)
        
        # Draw vertical lines for onset and cessation
        onset_date_val = future_df[future_df['dekad_of_year'] == onset_dekad]['date'].values
        cess_date_val = future_df[future_df['dekad_of_year'] == cessation_dekad]['date'].values
        
        if len(onset_date_val) > 0:
            onset_year = pd.to_datetime(onset_date_val[0]).year
            onset_label_str = dekad_to_date_str(onset_dekad, onset_year)
            ax.axvline(onset_date_val[0], color='#2e7d32', linestyle='--', linewidth=1.5, label=f"Onset ({onset_label_str})")
        if len(cess_date_val) > 0:
            cess_year = pd.to_datetime(cess_date_val[0]).year
            cess_label_str = dekad_to_date_str(cessation_dekad, cess_year)
            ax.axvline(cess_date_val[0], color='#c62828', linestyle='--', linewidth=1.5, label=f"Cessation ({cess_label_str})")
            
        ax.set_title('Decadal Precipitation Forecast with Agricultural Markers', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timeline')
        ax.set_ylabel('Rainfall Height (mm)')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')
        
        st.pyplot(fig)
        plt.close()
        
        # Crop Advisor Block
        st.markdown("---")
        st.write("### 🌾 Localized Crop Suitability & Planting Advisor")
        st.write("Based on the forecasted season length and total cumulative volume, here is the suitability assessment for common local crops:")
        
        # Standard crop durations in days
        crops = [
            {"name": "Maize (Early Maturing)", "duration": 90, "water": 500, "desc": "Good for shorter seasons. Can be planted right at onset."},
            {"name": "Maize (Late Maturing)", "duration": 120, "water": 700, "desc": "Requires a solid 4-month wet period. Yields high but vulnerable to early cessation."},
            {"name": "Rice (Upland)", "duration": 110, "water": 800, "desc": "Requires steady rain. Check for dry spells during flowering."},
            {"name": "Cassava", "duration": 270, "water": 1000, "desc": "Drought tolerant once established. Grows past cessation, harvesting in dry season."},
            {"name": "Yam", "duration": 210, "water": 1200, "desc": "Requires a long wet season. Planting mounds should be prepared pre-onset."},
            {"name": "Cowpea (Beans)", "duration": 75, "water": 350, "desc": "Short duration, highly suitable. Often intercropped or planted post-maize."}
        ]
        
        c_cards = st.columns(3)
        for idx, crop in enumerate(crops):
            col_idx = idx % 3
            
            # Determine suitability
            if season_length_days >= crop['duration'] and total_predicted_rain >= (crop['water'] * 0.7):
                suitability = "Highly Suitable"
                status_class = "status-suitable"
            elif season_length_days >= (crop['duration'] - 20) and total_predicted_rain >= (crop['water'] * 0.5):
                suitability = "Marginal Suitability"
                status_class = "status-marginal"
            else:
                suitability = "Not Recommended"
                status_class = "status-unsuitable"
                
            with c_cards[col_idx]:
                st.markdown(f"""
                    <div class="crop-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="font-size: 1.1rem; color: #333;">{crop['name']}</strong>
                            <span class="{status_class}">{suitability}</span>
                        </div>
                        <p style="margin: 8px 0 3px 0; font-size: 0.85rem; color: #555;">⏳ Growth Cycle: {crop['duration']} Days | 💧 Min Water: {crop['water']} mm</p>
                        <p style="margin: 0; font-size: 0.85rem; color: #888; font-style: italic;">{crop['desc']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
        # Explanatory Note Box
        st.markdown("""
            <div class="advisory-box">
                <h4>💡 Farmer Advisory Note</h4>
                <p>Ensure planting beds are prepared and pre-onset weeding is completed before the predicted onset date. If <b>Marginal Suitability</b> is indicated for your target crop, consider early-maturing seed varieties or coordinate supplementary irrigation options to protect yields against early cessation dry spells.</p>
            </div>
        """, unsafe_allow_html=True)
