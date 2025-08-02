import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load('models/final_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title('Supply Chain Emission Factor Prediction')

st.write("""
This application predicts the Supply Chain Emission Factors with Margins
based on input features.
""")

# Define input fields for the features
# You'll need to adjust these based on the actual features your model was trained on
# Refer to the X.columns output to get the correct feature names

# Example input fields (replace with your actual features)
substance = st.selectbox('Substance', [0, 1, 2, 3], format_func=lambda x: ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'][x])
unit = st.selectbox('Unit', [0, 1], format_func=lambda x: ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'][x])
emission_without_margins = st.number_input('Supply Chain Emission Factors without Margins', value=0.0)
margins = st.number_input('Margins of Supply Chain Emission Factors', value=0.0)
reliability_score = st.slider('DQ ReliabilityScore of Factors without Margins', 1, 5, 3)
temporal_correlation = st.slider('DQ TemporalCorrelation of Factors without Margins', 1, 3, 2)
geographical_correlation = st.slider('DQ GeographicalCorrelation of Factors without Margins', 1, 1, 1) # Based on unique values
technological_correlation = st.slider('DQ TechnologicalCorrelation of Factors without Margins', 1, 5, 3)
data_collection = st.slider('DQ DataCollection of Factors without Margins', 1, 1, 1) # Based on unique values
source = st.selectbox('Source', [0, 1], format_func=lambda x: ['Commodity', 'Industry'][x])

# Create a DataFrame from the input values
input_data = pd.DataFrame([[
    substance,
    unit,
    emission_without_margins,
    margins,
    reliability_score,
    temporal_correlation,
    geographical_correlation,
    technological_correlation,
    data_collection,
    source
]], columns=['Substance', 'Unit', 'Supply Chain Emission Factors without Margins',
       'Margins of Supply Chain Emission Factors',
       'DQ ReliabilityScore of Factors without Margins',
       'DQ TemporalCorrelation of Factors without Margins',
       'DQ GeographicalCorrelation of Factors without Margins',
       'DQ TechnologicalCorrelation of Factors without Margins',
       'DQ DataCollection of Factors without Margins', 'Source'])

# Ensure the order of columns matches the training data
# This is crucial if you used feature engineering that changed the column order
# If you used engineered features, you'll need to recreate them here based on the input data
# For simplicity, we are using the original features here. If you want to use engineered features,
# you'll need to apply the same PolynomialFeatures transformation to the input_data.

# Scale the input data
# If you used X_scaled_engineered for training, you should use the scaler_engineered here
# For simplicity, we are using the original scaler trained on X
scaled_input_data = scaler.transform(input_data)


if st.button('Predict Emission Factor'):
    # Make prediction
    prediction = model.predict(scaled_input_data)

    st.subheader('Prediction')
    st.write(f'Predicted Supply Chain Emission Factor with Margins: {prediction[0]:.4f}')
