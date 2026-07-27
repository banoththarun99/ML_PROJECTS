# Import Libraries
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import numpy as np

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_salary.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load Dataset
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded")

# Remove Unwanted Columns
remove_columns = [
    "survey_date",
    "recent_note"
    ]

for col in remove_columns:
    if col in df.columns:
        df.drop(col,axis=1,inplace=True)

# Features and Target
X = df.drop( "salary_bdt", axis=1)
y = df["salary_bdt"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.20,random_state=42)

# Load Model and Scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Scale Test Data
X_test = scaler.transform(X_test)

# Prediction
prediction = model.predict(X_test)

# Evaluation Metrics
r2 = r2_score( y_test, prediction)
mae = mean_absolute_error( y_test,prediction)
rmse = np.sqrt(mean_squared_error(y_test,prediction))
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print("R2 Score :", r2)
print("MAE      :", mae)
print("RMSE     :", rmse)

# Actual vs Predicted Plot
plt.figure(figsize=(8,5))
plt.scatter(y_test,prediction)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
plt.show()
plt.close()

# Residual Plot
residuals = y_test - prediction

plt.figure(figsize=(8,5))
plt.scatter(prediction,residuals)
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig(OUTPUT_DIR / "residual_plot.png")
plt.show()
plt.close()
print("\nGraphs Saved Successfully")