import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os

def build_path(*path_parts):
    return os.path.join(*path_parts)

# Load the trained model, Scaler, Label Encoder, OneHotEncoder, etc.
model = tf.keras.models.load_model(build_path('Model', 'model.h5'))

with open(build_path('Encoder', 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(build_path('Encoder', 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(build_path('Encoder', 'Scaler.pkl'), 'rb') as file:
    Scaler = pickle.load(file)

# Streamlit app UI
st.set_page_config(page_title='Bank Customer Churn Prediction', layout='wide')

# Custom CSS for sleek design
st.markdown("""
    <style>
        body {
            background-color: #f4f7f6;
            font-family: 'Arial', sans-serif;
        }
        .main {
            max-width: 1200px;
            margin: auto;
        }
        .sidebar .sidebar-content {
            background-color: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .header {
            font-size: 2em;
            color: #333;
        }
        .prediction-text {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 20px;
        }
        .prediction-yes {
            color: #e74c3c;
        }
        .prediction-no {
            color: #2ecc71;
        }
        .footer {
            font-size: 12px;
            color: #888;
            text-align: center;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Bank Customer Churn Prediction")
st.markdown("""
    Use the form below to input customer data and predict the likelihood of churn.
    Enter the information carefully to get the most accurate prediction.
""")

# Sidebar inputs
st.sidebar.header("Customer Information")
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92)
balance = st.sidebar.number_input('Balance', min_value=0)
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850)
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0)
tenure = st.sidebar.slider('Tenure', 0, 10)
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = Scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Main content area
st.markdown("---")
st.subheader("Prediction Results")

# Display churn probability
if prediction_proba > 0.5:
    st.markdown(f"<p class='prediction-text prediction-yes'>Churn Probability: {prediction_proba:.2f}</p>", unsafe_allow_html=True)
    st.write("**The customer is likely to churn.**")
else:
    st.markdown(f"<p class='prediction-text prediction-no'>Churn Probability: {prediction_proba:.2f}</p>", unsafe_allow_html=True)
    st.write("**The customer is not likely to churn.**")

# Add a "Learn More" button for explanations
if st.button('Learn More'):
    st.write("""
        The churn probability indicates the likelihood that a customer will leave the bank based on their profile.
        If the probability is higher than 0.5, the model predicts that the customer is likely to churn. 
        A probability below 0.5 means the customer is likely to stay.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <p class='footer'>
        This prediction is based on a machine learning model trained on historical customer data.
        Please note that the actual churn decision may vary based on other factors not included in the model.
    </p>
""", unsafe_allow_html=True)
