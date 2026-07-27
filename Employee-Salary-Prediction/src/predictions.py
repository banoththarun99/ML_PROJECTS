# Import Libraries
import pandas as pd
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# Load Model and Scaler
model = joblib.load( MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model Loaded Successfully")

# Employee Input
employee = {
    "age": 30,
    "gender": 1,
    "experience_years": 6,
    "skills_count": 8,
    "certifications": 3,
    "worked_remote": 1,
    "last_promotion_years_ago": 2,
    "recent_project_description_length": 60,

    "education_B.Sc": 0,
    "education_B.Sc+Cert": 0,
    "education_M.Eng": 0,
    "education_M.Sc": 1,
    "education_PhD": 0,

    "role_seniority_Junior": 0,
    "role_seniority_Lead": 0,
    "role_seniority_Mid": 1,
    "role_seniority_Senior": 0,

    "company_size_Enterprise": 1,
    "company_size_SME": 0,
    "company_size_Startup": 0,

    "location_tier_Remote": 0,
    "location_tier_Tier-1": 1,
    "location_tier_Tier-2": 0,
    "location_tier_Tier-3": 0
}

# Convert Input to DataFrame
input_data = pd.DataFrame([employee])

# Scaling
input_scaled = scaler.transform(input_data)

# Prediction
salary_prediction = model.predict(input_scaled)
print("=" * 50)
print("Predicted Salary:", round(salary_prediction[0],2),"BDT")
print("=" * 50)

