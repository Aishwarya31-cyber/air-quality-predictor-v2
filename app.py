import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load the AI Model and the Data
model = joblib.load('pm25_master_model.pkl')
df = pd.read_csv('pm25_data.xlsx - Sheet1.csv')

# 2. Page Design
st.set_page_config(page_title="Air Quality Dashboard", page_icon="🍃", layout="wide")
st.title("🍃 Predictive Air Quality Dashboard")

# 3. Sidebar: Input Controls
st.sidebar.header("🗓️ Prediction Timeline")
pred_date = st.sidebar.date_input("Select Date", datetime.date.today())
pred_time = st.sidebar.time_input("Select Time", datetime.datetime.now().time())

st.sidebar.markdown("---")
st.sidebar.header("☁️ Weather Parameters")
temp = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=10.0, step=1.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
wind_dir = st.sidebar.number_input("Wind Direction (° Degree)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("🚗 Human Factors")
traffic_map = {"Low (1)": 1, "Medium (2)": 2, "High (3)": 3}
traffic_choice = st.sidebar.selectbox("Traffic Volume", list(traffic_map.keys()))
traffic = traffic_map[traffic_choice]

industry_map = {"No Industrial Activity (0)": 0, "Industrial Area Nearby (1)": 1}
industry_choice = st.sidebar.selectbox("Industrial Presence", list(industry_map.keys()))
industry_near = industry_map[industry_choice]

# 4. Create the Tabs!
tab1, tab2 = st.tabs(["🎯 PM2.5 Prediction", "📊 Model Analytics"])

with tab1:
    st.write("Enter the local conditions in the sidebar to predict particulate matter levels.")
    
    if st.button("Predict PM2.5 Level", use_container_width=True):
        with st.spinner("Calculating PM2.5 based on inputs..."):
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

with tab2:
    st.header("AI Performance Metrics")
    st.write("This section proves the mathematical accuracy of our Machine Learning model.")
    
    # Calculate grades on the fly
    X = df[['wind_speed', 'wind_dir', 'humidity', 'temp', 'rainfall', 'traffic', 'industry_near']]
    y_actual = df['pm25']
    y_pred = model.predict(X)
    
    r2 = r2_score(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    # Display the Scores
    col1, col2 = st.columns(2)
    col1.metric("Accuracy (R² Score)", f"{r2:.2f}")
    col2.metric("Average Error (RMSE)", f"{rmse:.2f} µg/m³")
    
    st.markdown("---")
    
    # Graph 1: Actual vs Predicted
    st.subheader("📈 Actual vs. Predicted PM2.5")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.scatter(y_actual, y_pred, alpha=0.3, color='#1f77b4')
    ax1.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2) # The perfect line
    ax1.set_xlabel("Actual Real-World PM2.5")
    ax1.set_ylabel("AI Predicted PM2.5")
    st.pyplot(fig1)
    
    st.markdown("---")
    
    # Graph 2: Feature Importance (What the AI cares about most)
    st.subheader("🧠 What factors drive pollution?")
    importances = model.feature_importances_
    features = X.columns
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(feat_df['Feature'], feat_df['Importance'], color='#2ca02c')
    ax2.set_xlabel("Importance Level (0.0 to 1.0)")
    st.pyplot(fig2)