
# 📉 Bank Customer Churn Predictor

A machine learning application designed to predict customer churn in the banking sector. By analyzing customer data, this tool helps banks identify clients who are likely to leave, enabling proactive retention strategies.

#
### 📌 Table of Contents
- Overview
- Dataset
- Features
- Installation
- Usage
- Project Structure
- Technologies Used
- Contact

#
## 🧾 Overview
Customer churn is a significant concern for banks, impacting revenue and growth. This project leverages machine learning algorithms to predict the likelihood of a customer leaving the bank. By understanding the factors influencing churn, banks can implement targeted strategies to retain valuable customers.

## 📂 Dataset
- Source: Kaggle - Churn Modeling Dataset
- Description: The dataset contains information about bank customers, including       demographics, account information, and whether they have exited the bank.

 ***Key Features :***
- CustomerId: Unique identifier for each customer
- CreditScore: Credit score of the customer
- Geography: Customer's country of residence
- Gender: Gender of the customer
- Age: Age of the customer
- Tenure: Number of years the customer has been with the bank
- Balance: Account balance
- NumOfProducts: Number of bank products the customer uses
- HasCrCard: Whether the customer has a credit card
- IsActiveMember: Whether the customer is an active member
- EstimatedSalary: Estimated annual salary of the customer
- Exited: Whether the customer has left the bank (1) or not (0)

#
## ✨ Features
- Data Preprocessing: Handling missing values, encoding categorical variables, and feature scaling.
- Model Training: Utilizes classification algorithms to predict churn.
- Evaluation Metrics: Accuracy, precision, recall, and F1-score to assess model performance.
- Web Interface: Interactive web applications built with Flask and Streamlit for user-friendly predictions.

#
## 🛠️ Installation

**1. Clone the repository:**
```
   git clone https://github.com/Raj-UtsaV/Bank-Customer-Churn-Predictor.git
   cd Bank-Customer-Churn-Predictor
```

**2. Create and activate a virtual environment (optional but recommended):**
```
   python -m venv venv
```
*On Windows*
```
venv\Scripts\activate
```
*On Unix or MacOS*
```
source venv/bin/activate
```

**3. Install the required packages:**
```
   pip install -r requirements.txt
```

## 🚀 Usage
Using Streamlit Interface:
```
   streamlit run streamlit_app.py
```

Using Flask Interface:
```
   python Flask_app.py
```


#
## 📁 Project Structure
```
├── Data
│   ├── Churn_Modelling.csv
│   ├── X_test.csv
│   ├── X_train.csv
│   ├── y_test.csv
│   └── y_train.csv
├── Encoder
│   ├── label_encoder_gender.pkl
│   ├── onehot_encoder_geo.pkl
│   └── Scaler.pkl
├── Model
│   └── model.h5
├── notebooks
│   ├── Data_Preprocessing.ipynb
│   ├── Model_Training.ipynb
│   └── Prediction.ipynb
├── static
│   └── styles.css
├── templates
│  └── index.html
├── requirements.txt
├── Flask_app.py
└── Streamlit_app.py
```
#
## 🧰 Technologies Used
- Programming Language: Python
- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-learn
- Web Frameworks: Flask, Streamlit
- Visualization: Matplotlib, Seaborn
- Model Serialization: Pickle



#
## 📬 Contact
- Name: Utsav Raj
- LinkedIn: https://www.linkedin.com/in/utsav-raj-6657b12bb
- Email: utsavraj911@outlook.com
