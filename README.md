# Heart Disease Prediction System

A Machine Learning-based system that predicts the likelihood of heart disease using patient medical data.  
The project follows a complete ML pipeline including preprocessing, model comparison, evaluation, and deployment using Streamlit.

---

## Project Overview

This project builds a predictive system using supervised learning algorithms to classify whether a patient is at high risk or low risk of heart disease.

The system includes:
- Data preprocessing
- Multiple model training
- Model comparison
- Best model selection
- Real-time prediction UI

---

## Problem Statement

To develop a machine learning model that predicts the likelihood of heart disease based on patient medical attributes such as age, cholesterol level, chest pain type, etc.

The goal is to assist in early diagnosis using predictive analytics.

---

## Dataset

- Dataset: Heart Disease Dataset  
- Type: Tabular medical dataset  
- Features include:
  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Cholesterol
  - Fasting Blood Sugar
  - Rest ECG
  - Maximum Heart Rate
  - Exercise Induced Angina
  - Oldpeak
  - Slope
  - Number of Major Vessels
  - Thalassemia

Target Variable:
- 0 → No Heart Disease
- 1 → Heart Disease

---

## Solution Approach

The project follows a structured ML workflow:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Feature Scaling using StandardScaler
4. Train-Test Split (with Stratification)
5. Model Training:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - Artificial Neural Network (ANN - TensorFlow)
6. Model Evaluation using:
   - Accuracy
   - ROC-AUC Score
   - Confusion Matrix
   - Classification Report
7. Model Comparison
8. Best Model Selection
9. Model Saving using Joblib
10. Deployment using Streamlit

---

## Model Comparison

| Model                | Accuracy | ROC-AUC  |
|----------------------|----------|----------|
| Logistic Regression  | 0.80     | 0.87     |
| Random Forest        | 0.75     | 0.86     |
| SVM                  | 0.77     | 0.84     |
| ANN (TensorFlow)     | 0.77     | 0.86     |

---

## Final Model Selection

**Selected Model:** Logistic Regression  

Reason:
- Highest ROC-AUC score
- Stable performance
- Less overfitting compared to ANN
- Suitable for tabular medical data

The final model and scaler are saved as:

- `heart_disease_model.pkl`
- `scaler.pkl`

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow (for ANN experimentation)
- Matplotlib & Seaborn
- Streamlit
- Joblib

---

## Project Structure

---

Disease-Prediction/
│
├── app.py
├── heart_disease_model.pkl
├── scaler.pkl
├── Heart_DIsease2.ipynb
├── requirements.txt
└── README.md

---

## How to Run the Project

### Step 1: Install Dependencies

pip install -r requirements.txt

### step 2: Run the application

streamlit run app.py

The application will open in your browser where you can enter patient details and get prediction results.

---

## Conclusion

This project presents an end-to-end machine learning solution for heart disease prediction using patient medical data. Multiple models were evaluated, and Logistic Regression was selected based on ROC-AUC performance. The system includes preprocessing, model comparison, and a real-time Streamlit interface for practical deployment.

---