# Import Libraries
import pandas as pd
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "salary_dataset.csv"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_salary.csv"

# Load Dataset
df = pd.read_csv(RAW_DATA_PATH)

# Handle Missing Values
print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# Numerical columns
numeric_columns = df.select_dtypes(include="number").columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())
    
# Categorical columns
categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])
    
print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Dataset Information Before Cleaning
print("=" * 60)
print("DATASET INFORMATION BEFORE CLEANING")
print("=" * 60)

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nDataset Information")
df.info()

print("\nData Types")
print(df.dtypes)

# Missing Value Analysis
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

print("\nMissing Value Percentage")

missing_percentage = (missing_values / len(df)) * 100

print(missing_percentage.round(2))

# Duplicate Record Analysis
print("\n" + "=" * 60)
print("DUPLICATE RECORD ANALYSIS")
print("=" * 60)

duplicate_rows = df.duplicated().sum()

print(f"Duplicate Rows : {duplicate_rows}")

# Create Working Copy
df_clean = df.copy()

# Remove Duplicate Records
rows_before = len(df_clean)

df_clean = df_clean.drop_duplicates()

rows_after = len(df_clean)

print("\nDuplicate Records Removed Successfully")

print(f"Rows Before Cleaning : {rows_before}")
print(f"Rows After Cleaning  : {rows_after}")
print(f"Removed Rows         : {rows_before - rows_after}")

# Target Variable Analysis
target = "salary_bdt"

print("\n" + "=" * 60)
print("TARGET VARIABLE ANALYSIS")
print("=" * 60)

print(df_clean[target].describe())

print("\nMean Salary")
print(df_clean[target].mean())

print("\nMedian Salary")
print(df_clean[target].median())

print("\nMinimum Salary")
print(df_clean[target].min())

print("\nMaximum Salary")
print(df_clean[target].max())

print("\nStandard Deviation")
print(df_clean[target].std())

print("\nVariance")
print(df_clean[target].var())

print("\nSkewness")
print(df_clean[target].skew())

# Final Dataset Information
print("\n" + "=" * 60)
print("FINAL DATASET INFORMATION")
print("=" * 60)

print(f"Rows    : {df_clean.shape[0]}")
print(f"Columns : {df_clean.shape[1]}")



# Save Cleaned Dataset
df_clean.to_csv(
    CLEAN_DATA_PATH,
    index=False
)

print("\nCleaned dataset saved successfully.")
print(f"Location : {CLEAN_DATA_PATH}")