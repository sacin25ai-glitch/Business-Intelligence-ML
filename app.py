import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model
# -------------------------------

model = joblib.load("churn_model (2).pkl")

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------

st.title("📊 Business Intelligence Dashboard")

st.markdown("""
Predict whether a customer is likely to **Churn** based on customer details.
""")

st.divider()

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    1,
    72,
    12
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    20.0,
    120.0,
    60.0
)

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card"
    ]
)

# -------------------------------
# Encoding
# -------------------------------

gender_value = 1 if gender == "Male" else 0

contract_dict = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}

payment_dict = {
    "Electronic check":0,
    "Mailed check":1,
    "Bank transfer":2,
    "Credit card":3
}

contract_value = contract_dict[contract]
payment_value = payment_dict[payment]

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Churn"):

    data = [[
        gender_value,
        tenure,
        monthly,
        contract_value,
        payment_value
    ]]

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠ Customer is likely to Churn")

    else:

        st.success("✅ Customer is NOT likely to Churn")

    st.write("Probability")

    st.progress(float(max(probability[0])))

    st.write(probability)

st.divider()

# -------------------------------
# Dataset Preview
# -------------------------------

st.subheader("Dataset Preview")

df = pd.read_csv("business_intelligence_dataset.csv")

st.dataframe(df.head(20), use_container_width=True)

st.divider()

# -------------------------------
# Dataset Information
# -------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))

col2.metric(
    "Average Monthly Charges",
    round(df["MonthlyCharges"].mean(),2)
)

col3.metric(
    "Average Tenure",
    round(df["Tenure"].mean(),2)
)

st.divider()

# -------------------------------
# Churn Distribution
# -------------------------------

st.subheader("Customer Churn Distribution")

st.bar_chart(df["Churn"].value_counts())

# -------------------------------
# Monthly Charges
# -------------------------------

st.subheader("Monthly Charges")

st.line_chart(df["MonthlyCharges"])

# -------------------------------
# Tenure
# -------------------------------

st.subheader("Customer Tenure")

st.area_chart(df["Tenure"])

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.write("Developed using Python, Scikit-Learn and Streamlit")