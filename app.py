# ===============================
# Loan Default Prediction App
# ===============================

import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Default Prediction System")

st.write("Enter applicant details below")

# User Inputs
no_of_dependents = st.number_input("Number of Dependents", min_value=0)
education = st.selectbox("Education (Graduate=1, Not Graduate=0)", [0, 1])
self_employed = st.selectbox("Self Employed (Yes=1, No=0)", [0, 1])
income_annum = st.number_input("Annual Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=0)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0.0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0.0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0.0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0.0)

if st.button("Predict"):

    # Create engineered feature
    income_loan_ratio = income_annum / (loan_amount + 1)

    # Create DataFrame with correct column order
    input_data = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        income_loan_ratio
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.subheader("Result")

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")