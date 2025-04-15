import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model, Scaler, and encoders
def build_path(*path_parts):
    return os.path.join(os.getcwd(), *path_parts)

model = tf.keras.models.load_model(build_path('Model', 'model.h5'))

with open(build_path('Encoder', 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(build_path('Encoder', 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(build_path('Encoder', 'Scaler.pkl'), 'rb') as file:
    Scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input values from the form
        geography = request.form['geography']
        gender = request.form['gender']
        age = int(request.form['age'])
        balance = float(request.form['balance'])
        credit_score = int(request.form['credit_score'])
        estimated_salary = float(request.form['estimated_salary'])
        tenure = int(request.form['tenure'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])

        # Prepare the input data
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

        # Return prediction result
        result = 'The customer is likely to churn.' if prediction_proba > 0.5 else 'The customer is not likely to churn.'
        return render_template('index.html', churn_probability=f"{prediction_proba:.2f}", result=result)

    return render_template('index.html', churn_probability=None, result=None)

if __name__ == '__main__':
    app.run(debug=True)
