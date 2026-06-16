import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🌫 Global AQI Predictor", layout="centered")
st.title("🌫 Global AQI Predictor")

from tensorflow.keras.models import load_model

model = load_model("aqi_transformer_model.h5", compile=False)

# User Inputs

st.subheader("Enter Location & Pollutant Values:")

country = st.text_input("Country (any string):")
city = st.text_input("City (any string):")
lat = st.number_input("Latitude (-90 to 90):", min_value=-90.0, max_value=90.0, value=0.0)
lng = st.number_input("Longitude (-180 to 180):", min_value=-180.0, max_value=180.0, value=0.0)
co = st.number_input("CO AQI Value:", min_value=0.0, max_value=100.0, value=1.0)
ozone = st.number_input("Ozone AQI Value:", min_value=0.0, max_value=100.0, value=1.0)
no2 = st.number_input("NO2 AQI Value:", min_value=0.0, max_value=100.0, value=1.0)
pm25 = st.number_input("PM2.5 AQI Value:", min_value=0.0, max_value=100.0, value=1.0)


# Prediction

if st.button("Predict AQI"):
    # Simple encoding: replace with LabelEncoder mappings if needed
    country_id = 0
    city_id = 0

    # Create feature array
    X_input = np.array([[country_id, city_id, lat, lng, co, ozone, no2, pm25]])
    
    # Scale input
    scaler = StandardScaler()
    X_input_scaled = scaler.fit_transform(X_input)
    
    # Reshape for transformer/LSTM/CNN
    X_input_scaled = X_input_scaled.reshape((X_input_scaled.shape[0], X_input_scaled.shape[1], 1))

    # Make prediction
    pred = model.predict(X_input_scaled)
    st.success(f"Predicted AQI Value: {pred[0][0]:.2f}")