from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

# Load model + encoder
model = joblib.load("claimguard_v1_model.pkl")
encoder = joblib.load("claimguard_v1_encoder.pkl")

# Prepare SHAP explainer
explainer = shap.Explainer(model)

# Define input schema
class Claim(BaseModel):
    icd_code: str
    cpt_code: str
    payer: str
    provider_type: str

# Create API
app = FastAPI()

@app.post("/predict")
def predict(claim: Claim):
    # 1. Format input
    df = pd.DataFrame([claim.dict()])
    encoded = encoder.transform(df)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # 2. Predict risk
    risk = model.predict_proba(encoded)[0][1]

    # 3. Run SHAP
    shap_values = explainer(encoded_df)
    sv = shap_values[0]

    # 4. Extract top 3 contributing features
    contributions = list(zip(sv.feature_names, sv.values))
    top_reasons = sorted(contributions, key=lambda x: -x[1])[:3]
    reasons = [f"{feat} (+{val:.2f})" for feat, val in top_reasons]

    # 5. Return full response
    return {
        "denial_risk": round(float(risk), 2),
        "reasons": reasons
    }
