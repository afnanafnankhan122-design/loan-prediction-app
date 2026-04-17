import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Loan Approval Prediction App")

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
cibil = st.number_input("CIBIL Score")
residential = st.number_input("Residential Assets")
commercial = st.number_input("Commercial Assets")
luxury = st.number_input("Luxury Assets")
bank = st.number_input("Bank Assets")

education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if st.button("Predict"):
    try:
        data = np.array([[education, self_employed, income, loan_amount,
                          loan_term, cibil, residential, commercial,
                          luxury, bank]])

        st.write("Input Data:", data)

        data = scaler.transform(data)

        prediction = model.predict(data)

        st.write("Prediction Value:", prediction)

        if prediction[0] == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Rejected ❌")

    except Exception as e:
        st.error(f"Error: {e}")