from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# 1. Initialize the API
app = FastAPI(
    title="Credit Risk Prediction API",
    description="A Machine Learning API to predict loan default risk.",
    version="1.0"
)

# 2. Load the trained model
# Make sure 'credit_risk_model.pkl' is in the SAME folder as this file
try:
    model = joblib.load('credit_risk_model.pkl')
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model file not found. Please check the filename.")
    model = None

# 3. Define the Input Data Structure (Data Validation)
# REPLACE these fields with the EXACT features your model uses!
class LoanApplication(BaseModel):
    # Example features - UPDATE THESE:
    income: float
    loan_amount: float
    credit_score: int
    # Add other features here (e.g., age, employment_years, etc.)

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_risk(application: LoanApplication):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input data to a DataFrame (required for Scikit-Learn)
    # The keys in this dictionary must match your training columns exactly
    input_data = pd.DataFrame([application.dict()])
    
    # Make prediction
    # Assuming 0 = Low Risk (Pay back), 1 = High Risk (Default)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data).max()
    
    result = "High Risk (Default)" if prediction[0] == 1 else "Low Risk (Approved)"
    
    return {
        "prediction": result,
        "confidence_score": round(float(probability), 2),
        "input_data": application.dict()
    }

# 5. Root Endpoint (Health Check)
@app.get("/")
def home():
    return {"message": "Credit Risk API is running! Go to /docs to test it."}