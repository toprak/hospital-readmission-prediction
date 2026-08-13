import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. MINIMALIST PAGE CONFIGURATION (Centered layout for a clean look)
st.set_page_config(page_title="Readmission Risk", page_icon="🏥", layout="centered")

@st.cache_resource
def load_objects():
    model = joblib.load('data/processed/xgboost_champion_model.pkl')
    columns = joblib.load('data/processed/model_columns.pkl')
    scaler = joblib.load('data/processed/scaler.pkl') 
    return model, columns, scaler

try:
    model, model_columns, scaler = load_objects()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Model Calibration")
st.sidebar.markdown("""
<small>Adjust the threshold to calibrate the decision boundary according to hospital bed capacity.</small>
""", unsafe_allow_html=True)
custom_threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.85, 0.01)

# --- MAIN SCREEN (Minimalist) ---
st.title("🏥 Readmission Risk Prediction")
st.markdown("Evaluate 30-day readmission probability based on clinical metrics.")
st.divider() # Elegant divider line

st.subheader("Patient Data")

# Neatly organized input fields
col1, col2 = st.columns(2)

with col1:
    number_inpatient = st.slider("Inpatient Visits", 0, 21, 0)
    time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 3)
    num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)
    num_outpatient = st.slider("Outpatient Visits", 0, 10, 0)
    
with col2:
    num_medications = st.slider("Number of Medications", 1, 81, 15)
    num_lab_procedures = st.slider("Lab Procedures", 1, 132, 45)
    num_emergency = st.slider("Emergency Visits", 0, 10, 0)
    diabetes_med = st.selectbox("Diabetes Medication?", ["Yes", "No"])

st.divider()

# Full-width block button
if st.button("Calculate Risk", type="primary", use_container_width=True):
    
    # Initialize dataframe with zeros
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Injecting healthy/normal default values to prevent the 'Medical Crisis' bug
    for col in model_columns:
        if isinstance(col, str):
            if any(keyword in col for keyword in ['_No', '_None', '_Norm', '_Steady', 'Female', 'Caucasian']):
                input_data.at[0, col] = 1
            
    # Scaler operations (Creating a temporary dataframe with all training columns)
    scaler_cols = scaler.feature_names_in_
    temp_scale_df = pd.DataFrame(np.zeros((1, len(scaler_cols))), columns=scaler_cols)
    
    # Assigning user inputs
    if 'time_in_hospital' in scaler_cols: temp_scale_df.at[0, 'time_in_hospital'] = time_in_hospital
    if 'num_lab_procedures' in scaler_cols: temp_scale_df.at[0, 'num_lab_procedures'] = num_lab_procedures
    if 'num_medications' in scaler_cols: temp_scale_df.at[0, 'num_medications'] = num_medications
    if 'number_inpatient' in scaler_cols: temp_scale_df.at[0, 'number_inpatient'] = number_inpatient
    if 'number_diagnoses' in scaler_cols: temp_scale_df.at[0, 'number_diagnoses'] = num_diagnoses
    if 'number_emergency' in scaler_cols: temp_scale_df.at[0, 'number_emergency'] = num_emergency
    if 'number_outpatient' in scaler_cols: temp_scale_df.at[0, 'number_outpatient'] = num_outpatient
    
    # Transforming and transferring the scaled data
    scaled_values = scaler.transform(temp_scale_df)
    scaled_df = pd.DataFrame(scaled_values, columns=scaler_cols)
    
    for col in ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient', 'number_diagnoses', 'number_emergency', 'number_outpatient']:
        if col in model_columns:
            input_data.at[0, col] = scaled_df.at[0, col]
    
    # Explicitly checking diabetes medication status
    if diabetes_med == "Yes" and 'diabetesMed_Yes' in model_columns:
        input_data.at[0, 'diabetesMed_Yes'] = 1
        
    with st.spinner("Analyzing..."):
        raw_probability = model.predict_proba(input_data)[0][1]
        
    # Displaying results neatly
    if raw_probability > custom_threshold:
        st.error(f"⚠️ **HIGH RISK:** Probability of readmission within 30 days is **{raw_probability * 100:.1f}%**")
    else:
        st.success(f"✅ **LOW RISK:** Probability of readmission within 30 days is **{raw_probability * 100:.1f}%**")