import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction Dashboard", "Model Information"])

st.sidebar.markdown("---")
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input( 
    "API Endpoint", 
    value="https://churn-api.onrender.com",
    help="Your FastAPI endpoint URL" 
)

st.sidebar.markdown("---")
st.sidebar.write("### Instructions")
if page == "Prediction Dashboard":
    st.sidebar.write("1. Fill in the customer details in the dashboard.")
    st.sidebar.write("2. Click 'Predict' to see if the customer is likely to churn.")
else:
    st.sidebar.write("Explore the model's performance and status details.")

# --- PREDICTION PAGE ---
if page == "Prediction Dashboard":
    st.title("📊 Customer Churn Prediction System")
    st.subheader("Predict customer churn using ML and behavioral data")
    
    # Customer Data Input
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Basic Info")
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months)", 0, 100, 24)
        monthly_charges = st.number_input("Monthly Charges ($)", value=75.5)
        total_charges = st.number_input("Total Charges ($)", value=1812.0)
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    with col3:
        st.subheader("Billing")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ])

    # Prepare data for API with all 30 features
    input_data = {
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender_Male": 1 if gender == "Male" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
        "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
        "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0
    }

    if st.button("Predict"):
        try:
            response = requests.post(f"{api_url}/predict", json=input_data)
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                probability = result['probability']
                confidence = result['confidence']
                message = result['message']
                
                if prediction == 1:
                    st.error(f"Prediction: **Churn**")
                else:
                    st.success(f"Prediction: **No Churn**")
                    
                st.write(f"**Confidence:** {confidence}")
                st.write(f"**Probability:** {probability:.2%}")
                st.info(f"**Details:** {message}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- MODEL INFORMATION PAGE ---
elif page == "Model Information":
    st.title("📊 Model Information")
    st.write("Strictly model performance metrics and system status.")
    st.divider()
    
    with st.spinner("Fetching model details from API..."):
        try:
            health_response = requests.get(f"{api_url}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info("### System Status")
                    st.metric("API Status", "🟢 Online", help="API is responding")
                    st.metric("Model Architecture", health_data.get('model', "N/A"))
                    st.metric("API Version", health_data.get('version', "N/A"))
                
                with col2:
                    st.info("### Performance Metrics")
                    st.metric("Accuracy", "91.4%")
                    st.metric("ROC-AUC", "0.89")
                    st.metric("Features Used", health_data.get('features', 0))
                
                st.divider()
                st.info("### Dataset Information")
                st.write("- **Total Customers in Training:** 7,043")
                st.write("- **Target Variable:** Churn (Yes/No)")
                st.write("- **Feature Engineering:** One-hot encoding, feature scaling applied.")
                
            else:
                st.error("API Status: Offline - Unable to load real-time model info")
        except Exception as e:
            st.error(f"API Connection Error: {e}")
            st.write("Please check if the API Endpoint in the sidebar is correct.")
