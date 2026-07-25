import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction System")
st.subheader("Created By Ravi Kumar Singh")
st.markdown("---")

model = joblib.load("car_price_model.pkl")
encoders = joblib.load("label_encoders.pkl")

year = st.number_input("Manufacturing Year", min_value=2000, max_value=2025, value=2020)
present_price = st.number_input("Present Price (Lakhs)", min_value=0.0, value=10.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=20000)

fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel"])
seller_type = st.selectbox("Seller Type", ["Dealer","Individual"])
transmission = st.selectbox("Transmission", ["Manual","Automatic"])
owner = st.selectbox("Owner", [0,1,2,3])

if st.button("Predict Car Price"):
    fuel = encoders["Fuel_Type"].transform([fuel_type])[0]
    seller = encoders["Seller_Type"].transform([seller_type])[0]
    trans = encoders["Transmission"].transform([transmission])[0]

    data = np.array([[year,present_price,kms_driven,fuel,seller,trans,owner]])
    prediction = model.predict(data)

    st.success(f"Predicted Selling Price: ₹ {prediction[0]:.2f} Lakhs")

st.markdown("---")
st.caption("Machine Learning Regression Model | Streamlit Deployment")
