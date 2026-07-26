# Kogi State Agrometeorological Decision Support System (ADSS)

This repository contains time-series analysis models and an interactive crop planning web application designed to forecast rainfall patterns and optimize crop planting calendars for farmers and extension officers in Kogi State, Nigeria.

The project models historical dekadal (10-day) rainfall data from CHIRPS (1981–2026) using classical statistical and machine learning architectures, and translates precipitation predictions into actionable crop calendars (onset dates, cessation dates, and suitability advisories).

---

## 🚀 Key Features

*   **Multimodel Forecasting**:
    *   **SARIMAX**: A statistical seasonal time-series model.
    *   **Random Forest Regressor**: A machine learning model using recursive lag-sliding windows.
    *   **LSTM (Long Short-Term Memory)**: A deep learning sequential neural network built in PyTorch.
*   **Agricultural Intelligence**:
    *   Dynamic calculation of **Rainfall Onset** (planting start) and **Cessation** (season end) using Walter/Benoit logical rules.
    *   Localized suitability advisor for popular Nigerian crops (**Maize, Rice, Yam, Cassava, and Cowpeas**) based on projected season lengths and cumulative water indices.
*   **Premium Streamlit Dashboard**:
    *   Interactive forecast horizon slider.
    *   Dynamic plot visualizer showing historical CHIRPS data aligned with predicted precipitation curves.
    *   Explainable AI (XAI) descriptions for each model.

---

## 📁 Repository Structure

*   `data_pipeline.py`: Loads the raw dataset, builds temporal/cyclical features, engineers lag arrays (past 36 dekads), and partitions chronological splits.
*   `crop_planning.py`: Houses the mathematical logic for onset, cessation, and season length calendar calculations.
*   `model_sarima.py`: Fits and serializes the Seasonal ARIMA model.
*   `model_random_forest.py`: Implements Random Forest training and recursive multi-step forecasting.
*   `model_lstm.py`: Implements PyTorch LSTM sequence neural network with epoch-based validation checkpointing.
*   `evaluate_models.py`: Runs all model training, performs blind testing, exports evaluation metrics, and saves the comparison plot.
*   `app.py`: The entry point for the Streamlit web dashboard.
*   `kogi_chirps_monthly.csv`: The historical subnational dekadal rainfall series extracted for Kogi State.

---

## ⚙️ Setup and Installation

### 1. Install Dependencies
Clone the repository and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Train the Models
To fit the SARIMAX, Random Forest, and LSTM models and output error metrics:
```bash
python evaluate_models.py
```
This will train the architectures, generate evaluation logs, save the models, and create `rainfall_forecast_comparison.png`.

### 3. Launch the Streamlit App
Start the interactive dashboard locally:
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 📈 Performance Benchmarks (Blind Test Set)

Models were evaluated out-of-sample on recent blind records:

| Model | RMSE | MAE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **LSTM (PyTorch)** | **18.95 mm** | **11.96 mm** | **0.635** |
| **SARIMAX** | 19.17 mm | 12.27 mm | 0.627 |
| **Random Forest** | 19.48 mm | 12.52 mm | 0.615 |
