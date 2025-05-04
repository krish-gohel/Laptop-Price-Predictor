import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Input UI
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight of the Laptop (kg)')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.3)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())

# Get HDD/SSD options with values up to 1TB (1000GB)
hdd_values = sorted(df['HDD'].fillna(0).astype(int).unique())
# Add common HDD sizes up to 1TB if not already present
common_hdd = [0, 128, 256, 512, 1000]  # Common HDD sizes including 1TB (1000GB)
hdd_values = sorted(list(set(hdd_values + common_hdd)))
hdd = st.selectbox('HDD (in GB)', hdd_values)

ssd_values = sorted(df['SSD'].fillna(0).astype(int).unique())
# Add common SSD sizes up to 1TB if not already present
common_ssd = [0, 128, 256, 512, 1000]  # Common SSD sizes including 1TB (1000GB)
ssd_values = sorted(list(set(ssd_values + common_ssd)))
ssd = st.selectbox('SSD (in GB)', ssd_values)

gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# Prediction
if st.button('Predict Price'):
    try:
        # Process binary and numerical inputs
        touchscreen_bin = 1 if touchscreen == 'Yes' else 0
        ips_bin = 1 if ips == 'Yes' else 0
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

        # Create input DataFrame
        input_df = pd.DataFrame([{
            'Company': company,
            'TypeName': type,
            'Ram': ram,
            'Weight': weight,
            'Touchscreen': touchscreen_bin,
            'IPS': ips_bin,
            'PPI': ppi,
            'Cpu brand': cpu,
            'HDD': hdd,
            'SSD': ssd,
            'Gpu brand': gpu,
            'os': os
        }])

        # Predict
        prediction = pipe.predict(input_df)[0]
        final_price = int(np.exp(prediction))

        st.success(f"The predicted price of this configuration is ₹{final_price}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
