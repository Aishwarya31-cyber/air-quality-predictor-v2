import streamlit as st
import pandas as pd
import joblib
import datetime

# 1. Load the Master AI Model
model = joblib.load('pm25_master_model.pkl')

# 2. Page Design
st.set_page_config(page_title="Air Quality Dashboard", page_icon="🍃", layout="centered")
st.title("🍃 Predictive Air Quality Dashboard")
st.write("Enter the local conditions below to predict particulate matter levels.")

# 3. Date & Time Selection
st.sidebar.header("🗓️ Prediction Timeline")
pred_date = st.sidebar.date_input("Select Date", datetime.date.today())
pred_time = st.sidebar.time_input("Select Time", datetime.datetime.now().time())

st.sidebar.markdown("---")

# 4. Weather Text Boxes
st.sidebar.header("☁️ Weather Parameters")
temp = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
wind_dir = st.sidebar.number_input("Wind Direction (° Degree)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)

st.sidebar.markdown("---")

# 5. Human Activity Dropdowns
st.sidebar.header("🚗 Human Factors")
traffic_map = {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3}
traffic_choice = st.sidebar.selectbox("Traffic Volume", list(traffic_map.keys()))
traffic = traffic_map[traffic_choice]

industry_map = {"No Industrial Activity (0)": 0, "Industrial Area Nearby (1)": 1}
industry_choice = st.sidebar.selectbox("Industrial Presence", list(industry_map.keys()))
industry_near = industry_map[industry_choice]

# 6. Make the Prediction
if st.button("Predict PM2.5 Level", use_container_width=True):
    with st.spinner("Calculating PM2.5 based on inputs..."):
        
        # BULLETPROOF FIX: These columns match the AI's brain perfectly!
        input_data = pd.DataFrame(
            [[wind_speed, wind_dir, humidity, temp, rainfall, traffic, industry_near]], 
            columns=['wind_speed', 'wind_dir', 'humidity', 'temp', 'rainfall', 'traffic', 'industry_near']
        )
        
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader(f"Forecast for {pred_date.strftime('%B %d, %Y')} at {pred_time.strftime('%I:%M %p')}")
        
        if prediction < 30:
            st.success(f"🟢 **{prediction:.2f} µg/m³** (Good Air Quality)")
        elif prediction < 60:
            st.warning(f"🟡 **{prediction:.2f} µg/m³** (Moderate Air Quality)")
        else:
            st.error(f"🔴 **{prediction:.2f} µg/m³** (Unhealthy Air Quality)")