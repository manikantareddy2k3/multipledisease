# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 09:20:24 2024

@author: dell
"""

import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Set the page configuration first
st.set_page_config(page_title="Disease Prediction", page_icon=":hospital:", layout="wide")

# Loading the saved models
cvd_model = pickle.load(open('Cvd_model.sav','rb'))
diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'CVD Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart'],
                           default_index=0,
                           styles={
                               "nav-link": {"font-size": "16px", "text-align": "center", "margin": "10px"},
                               "nav-item": {"padding": "8px"},
                               "container": {"padding": "5px"}
                           })

# Global UI improvements
# Title of the App
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Multiple Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #32CD32;'>Predict Diabetes &  CVD Using Machine Learning</h3>", unsafe_allow_html=True)
st.markdown("---")

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using Ensemble ML')

    # Styled Inputs for Diabetes prediction
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input('Age of the Person (Years)', min_value=0, step=1, help="Enter the age of the person.")
        BMI = st.number_input('BMI value (kg/m²)', min_value=0.0, step=0.1, help="Enter the BMI value. Normal BMI range: 18.5 to 24.9")

    with col2:
        HbA1c = st.number_input('HbA1c Level (%)', min_value=0.0, step=0.1, help="Normal range: 4.0 to 5.6%")
        Chol = st.number_input('Cholesterol Level (mmol/L)', min_value=0.0, step=0.1, help="Normal range: 3.6 to 5.2 mmol/L")

    # Prediction button with styled button
    if st.button('Predict Diabetes', key='predict_diabetes', use_container_width=True):
        input_data = pd.DataFrame([[Age, BMI, HbA1c, Chol]], columns=['AGE', 'BMI', 'HbA1c', 'Chol'])
        prediction = diabetes_model.predict(input_data)[0]
        prediction_prob = diabetes_model.predict_proba(input_data)[:, 1][0]

        if prediction == 1:
            st.success(f'The person is diabetic (Confidence: {prediction_prob:.2f})')
        else:
            st.success(f'The person is not diabetic (Confidence: {1 - prediction_prob:.2f})')

# Heart Disease Prediction Page
if selected == 'CVD Prediction':
    st.title('Cardiovascular Disease (CVD) Prediction using Ensemble ML')

    # Styled Inputs for CVD prediction
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age (Years)', min_value=0, step=1, help="Enter the age of the person.")
        restingBP = st.number_input('Resting Blood Pressure (mmHg)', min_value=0, step=1, help="Enter the resting blood pressure.")

    with col2:
        serum_cholesterol = st.number_input('Serum Cholesterol (mg/dL)', min_value=0, step=1, help="Enter the serum cholesterol level.")
        gender = st.selectbox('Gender', options=['Male', 'Female'], help="Select the gender.")

    gender_numeric = 1 if gender == 'Male' else 0

    # Prediction button with styled button
    if st.button('Predict Heart Disease', key='predict_cvd', use_container_width=True):
        input_data = pd.DataFrame([[age, gender_numeric, restingBP, serum_cholesterol]], 
                                  columns=['AGE', 'Gender', 'restingBP', 'serumcholestrol'])
        prediction = cvd_model.predict(input_data)[0]
        prediction_prob = cvd_model.predict_proba(input_data)[:, 1][0]

        if prediction == 1:
            st.success(f'The person is at risk of heart disease (Confidence: {prediction_prob:.2f})')
        else:
            st.success(f'The person is not at risk of heart disease (Confidence: {1 - prediction_prob:.2f})')
