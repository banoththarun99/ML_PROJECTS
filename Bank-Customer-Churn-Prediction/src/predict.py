# predict.py

import joblib
import pandas as pd

# Load Saved Model
model = joblib.load(
    "models/random_forest.pkl"
)

# New Customer Data
new_customer = pd.DataFrame([{
    "CreditScore": 650,
    "Gender": 1,
    "Age": 45,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 70000,
    "Satisfaction Score": 3,
    "Point Earned": 500,

    "Geography_France": 0,
    "Geography_Germany": 1,
    "Geography_Spain": 0,

    "Card Type_DIAMOND": 0,
    "Card Type_GOLD": 1,
    "Card Type_PLATINUM": 0,
    "Card Type_SILVER": 0
}])

# Prediction
prediction = model.predict(new_customer)

# Result
if prediction[0] == 1:
    print("Customer Will Churn")
else:
    print("Customer Will Stay")
    
probability = model.predict_proba(new_customer)

print("\nPrediction Probabilities:")
print(probability)