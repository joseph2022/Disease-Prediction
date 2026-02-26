import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor")

st.title(" Heart Disease Prediction System")
st.markdown("Fill the details below to check heart disease risk.")

st.markdown("---")

# Input Section 

age = st.number_input("Age (20 - 100 years)", min_value=20, max_value=100, value=50)

sex = st.radio("Sex", ["Male", "Female"])

cp = st.selectbox(
    "Chest Pain Type",
    {
        0: "0 - Typical Angina",
        1: "1 - Atypical Angina",
        2: "2 - Non-anginal Pain",
        3: "3 - Asymptomatic"
    }
)

trestbps = st.number_input(
    "Resting Blood Pressure (80 - 200 mm Hg)",
    min_value=80, max_value=200, value=120
)

chol = st.number_input(
    "Cholesterol (100 - 600 mg/dL)",
    min_value=100, max_value=600, value=200
)

fbs = st.radio(
    "Fasting Blood Sugar > 120 mg/dL",
    {0: "No", 1: "Yes"}
)

restecg = st.selectbox(
    "Resting ECG Results",
    {
        0: "0 - Normal",
        1: "1 - ST-T Wave Abnormality",
        2: "2 - Left Ventricular Hypertrophy"
    }
)

thalach = st.number_input(
    "Maximum Heart Rate (70 - 210)",
    min_value=70, max_value=210, value=150
)

exang = st.radio(
    "Exercise Induced Angina",
    {0: "No", 1: "Yes"}
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak) (0.0 - 6.0)",
    min_value=0.0, max_value=6.0, value=1.0
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    {
        0: "0 - Upsloping",
        1: "1 - Flat",
        2: "2 - Downsloping"
    }
)

ca = st.selectbox(
    "Number of Major Vessels (0 - 3)",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    {
        1: "1 - Normal",
        2: "2 - Fixed Defect",
        3: "3 - Reversible Defect"
    }
)

st.markdown("---")


# Prediction


if st.button("Predict Heart Disease Risk"):

    sex_value = 1 if sex == "Male" else 0

    input_data = [[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]]

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    input_df = pd.DataFrame(input_data, columns=columns)
    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("##  Prediction Result")

    if probability > 0.5:
        st.error(f" HIGH RISK of Heart Disease ({round(probability*100,2)}%)")
    else:
        st.success(f" LOW RISK ({round(probability*100,2)}%)")