import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_model.pkl')  # Load the trained model

st.title("Fraud Detection Prediction")

st.markdown("Please enter the details for prediction:")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH _OUT", "DEBIT", "CASH_IN"])
amount = st.number_input("Amount", min_value=0.0, value = 1000.0)
oldbalanceOrg = st.number_input("Old Balance Sender", min_value=0.0, value = 10000.0)
newbalanceOrig = st.number_input("New Balance Sender", min_value=0.0, value = 9000.0)
oldbalanceDest = st.number_input("Old Balance Receiver", min_value=0.0, value = 0.0)
newbalanceDest = st.number_input("New Balance Receiver", min_value=0.0, value = 0.0)

if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'type': [transaction_type],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    # Make prediction
    prediction = model.predict(input_data)

    st.subheader("Prediction Result: ")

    if prediction[0] == 1:
        st.success("The transaction is predicted to be fraudulent.")
    else:
        st.success("The transaction is predicted to be legitimate.")