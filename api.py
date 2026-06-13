from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using Machine Learning",
    version="2.0.0"
)

MODEL_PATH = 'models/random_forest_tuned.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURE_NAMES_PATH = 'data/processed/feature_names.txt'

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
        model = None

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Scaler loaded from {SCALER_PATH}")
    else:
        print(f"⚠️ Scaler file not found at {SCALER_PATH}")
        scaler = None

    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, 'r') as f:
            FEATURE_NAMES = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(FEATURE_NAMES)} feature names")
    else:
        print(f"⚠️ Feature names file not found at {FEATURE_NAMES_PATH}")
        FEATURE_NAMES = []

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    model = None
    scaler = None
    FEATURE_NAMES = []

class CustomerData(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1)
    tenure: float = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    gender_Male: int = Field(..., ge=0, le=1)
    Partner_Yes: int = Field(..., ge=0, le=1)
    Dependents_Yes: int = Field(..., ge=0, le=1)
    PhoneService_Yes: int = Field(..., ge=0, le=1)
    MultipleLines_No_phone_service: int = Field(..., ge=0, le=1, alias="MultipleLines_No phone service")
    MultipleLines_Yes: int = Field(..., ge=0, le=1)
    InternetService_Fiber_optic: int = Field(..., ge=0, le=1, alias="InternetService_Fiber optic")
    InternetService_No: int = Field(..., ge=0, le=1)
    OnlineSecurity_No_internet_service: int = Field(..., ge=0, le=1, alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int = Field(..., ge=0, le=1)
    OnlineBackup_No_internet_service: int = Field(..., ge=0, le=1, alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int = Field(..., ge=0, le=1)
    DeviceProtection_No_internet_service: int = Field(..., ge=0, le=1, alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int = Field(..., ge=0, le=1)
    TechSupport_No_internet_service: int = Field(..., ge=0, le=1, alias="TechSupport_No internet service")
    TechSupport_Yes: int = Field(..., ge=0, le=1)
    StreamingTV_No_internet_service: int = Field(..., ge=0, le=1, alias="StreamingTV_No internet service")
    StreamingTV_Yes: int = Field(..., ge=0, le=1)
    StreamingMovies_No_internet_service: int = Field(..., ge=0, le=1, alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: int = Field(..., ge=0, le=1)
    Contract_One_year: int = Field(..., ge=0, le=1, alias="Contract_One year")
    Contract_Two_year: int = Field(..., ge=0, le=1, alias="Contract_Two year")
    PaperlessBilling_Yes: int = Field(..., ge=0, le=1)
    PaymentMethod_Credit_card_automatic: int = Field(..., ge=0, le=1, alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(..., ge=0, le=1, alias="PaymentMethod_Mailed check")

    class Config:
        allow_population_by_field_name = True

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=No Churn, 1=Churn")
    probability: float = Field(..., description="Probability of churn (0-1)")
    confidence: str = Field(..., description="High/Medium/Low confidence")
    message: str = Field(..., description="Interpretation message")

@app.get("/health", tags=["Health"])
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    if len(FEATURE_NAMES) != 30:
        raise HTTPException(status_code=503, detail="Feature names not loaded correctly")

    return {
        "status": "healthy",
        "model": "Random Forest (Tuned)",
        "version": "2.0.0",
        "features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    if len(FEATURE_NAMES) != 30:
        raise HTTPException(status_code=503, detail="Feature names not loaded")

    try:
        customer_dict = customer.model_dump(by_alias=True)
        X = pd.DataFrame([customer_dict])
        X = X[FEATURE_NAMES]
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)

        prediction = int(model.predict(X_scaled_df)[0])
        probability = float(model.predict_proba(X_scaled_df)[0][1])

        confidence_score = max(model.predict_proba(X_scaled_df)[0])
        if confidence_score > 0.8:
            confidence = "High"
        elif confidence_score > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        if prediction == 1:
            message = f"Customer is likely to CHURN (probability: {probability:.2%})"
        else:
            message = f"Customer is likely to STAY (probability: {1-probability:.2%})"

        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
