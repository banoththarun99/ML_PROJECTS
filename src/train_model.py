# importing libraries
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_salary.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
(BASE_DIR / "models").mkdir(exist_ok=True)
 
# Load Data
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded")
print(df.head())

print("\nDATA CHECK IN TRAIN MODEL")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove Unwanted Columns
remove_columns = ["survey_date","recent_note"]
for col in remove_columns:
    if col in df.columns:
        df.drop(col,axis=1,inplace=True)

# Features and Target
X = df.drop("salary_bdt",axis=1)
y = df["salary_bdt"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform( X_test)
joblib.dump(
    scaler,
    SCALER_PATH
)
print("Scaler Saved")

# Models
models = {
"Linear Regression":
LinearRegression(),
"Decision Tree":
DecisionTreeRegressor(
    random_state=42
),
"Random Forest":
RandomForestRegressor(
    n_estimators=100,
    random_state=42
),
"XGBoost":
XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)
}

# Training
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import numpy as np


# Training
results = {}
trained_models = {}

for name, model in models.items():

    print("\nTraining", name)

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )


    r2 = r2_score(
        y_test,
        prediction
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    results[name] = {
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }


    trained_models[name] = model


    print("R2 Score :", r2)
    print("MAE      :", mae)
    print("RMSE     :", rmse)
    
# Select Best Model based on R2 Score
best_model_name = max(
    results,
    key=lambda x: results[x]["R2 Score"]
)

best_model = trained_models[best_model_name]

print("\nBest Model:")
print(best_model_name)
# Save Model
joblib.dump(
    best_model,
    MODEL_PATH
)
print("\nModel Saved Successfully")