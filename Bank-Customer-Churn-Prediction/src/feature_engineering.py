# Import Libraries
import pandas as pd

# Load Cleaned Dataset
df = pd.read_csv("venv/Bank-Customer-Churn-Prediction/data/cleaned_churn.csv")

# Create Backup Copy
df_backup = df.copy()

# Create Working Copy
df_encoded = df.copy()

# Shape Before Encoding
print("Shape Before Encoding:")
print(df_encoded.shape)

# Binary Encoding
df_encoded["Gender"] = df_encoded["Gender"].map({"Male": 1,"Female": 0})

# One-Hot Encoding
df_encoded = pd.get_dummies(df_encoded,columns=["Geography", "Card Type"],dtype=int)

# Shape After Encoding
print("\nShape After Encoding:")
print(df_encoded.shape)

# Display Columns
print("\nColumns After Encoding:")
print(df_encoded.columns.tolist())

# Save Processed Dataset
df_encoded.to_csv("venv/Bank-Customer-Churn-Prediction/data/processed_churn.csv",index=False)


