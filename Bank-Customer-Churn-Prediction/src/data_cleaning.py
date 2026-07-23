# Import Libraries
import pandas as pd

# Load Dataset
df = pd.read_csv("venv/Bank-Customer-Churn-Prediction/data/Customer_Churn_Records.csv")

# Dataset Information Before Cleaning
print("Original Shape:")
print(df.shape)
print()

print("Original Columns:")
print(df.columns.tolist())
print()

# Create Backup Copy
df_backup = df.copy()


# Create Working Copy
df_clean = df.copy()

# Remove Unnecessary Columns
unnecessary_columns = ["RowNumber", "CustomerId", "Surname"]
df_clean = df_clean.drop(columns=unnecessary_columns)
print("Columns Removed Successfully")
print()

# Dataset Information After Cleaning
print("Cleaned Shape:")
print(df_clean.shape)
print()

print("Remaining Columns:")
print(df_clean.columns.tolist())
print()

# Save Cleaned Dataset
df_clean.to_csv("venv/Bank-Customer-Churn-Prediction/data/cleaned_churn.csv",index=False)