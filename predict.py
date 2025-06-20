import joblib  # 1
import pandas as pd  # 2

# 3. Load the model and encoder
model = joblib.load("claimguard_v1_model.pkl")
encoder = joblib.load("claimguard_v1_encoder.pkl")

# 4. Sample claim to test
claim = {
    "icd_code": "S43.421A",
    "cpt_code": "73721",
    "payer": "Blue Cross",
    "provider_type": "Chiropractor"
}

# 5. Format as DataFrame
df = pd.DataFrame([claim])

# 6. Encode using the saved encoder
encoded = encoder.transform(df)
feature_names = encoder.get_feature_names_out(df.columns)

# 7. Predict denial risk
risk = model.predict_proba(encoded)[0][1]

# 8. Print result
print(f"🧠 Denial Risk: {risk:.2f}")
