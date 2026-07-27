# Import Libraries
import pandas as pd
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_salary.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_salary.csv"

# Load Cleaned Dataset
df = pd.read_csv(CLEAN_DATA_PATH)

# Create Backup Copy
df_backup = df.copy()

# Create Working Copy
df_encoded = df.copy()

# Shape Before Encoding
print("=" * 60)
print("SHAPE BEFORE ENCODING")
print("=" * 60)

print(df_encoded.shape)

# Binary Encoding
df_encoded["gender"] = df_encoded["gender"].map({
    "Male": 1,
    "Female": 0
})

# Handle unknown values created during encoding
df_encoded["gender"] = df_encoded["gender"].fillna(
    df_encoded["gender"].mode()[0]
)

# One-Hot Encoding
categorical_columns = [
    "education",
    "role_seniority",
    "company_size",
    "location_tier"
]

df_encoded = pd.get_dummies(
    df_encoded,
    columns=categorical_columns,
    dtype=int
)

# Shape After Encoding
print("\n" + "=" * 60)
print("SHAPE AFTER ENCODING")
print("=" * 60)
print(df_encoded.shape)

# Columns After Encoding
print("\nColumns After Encoding")
print(df_encoded.columns.tolist())

print("\nMissing Values Before Saving:")
print(df_encoded.isnull().sum())

# Save Processed Dataset
df_encoded.to_csv(
    PROCESSED_DATA_PATH,
    index=False
)

print("\nProcessed dataset saved successfully.")
print("Location :", PROCESSED_DATA_PATH)

