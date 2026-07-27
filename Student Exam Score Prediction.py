#  Taking user input
hours = float(input("Enter Hours Studied: "))
sleep = float(input("Enter Sleep Hours: "))
attendance = float(input("Enter Attendance: "))


#  Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


#  Loading dataset
df = pd.read_csv("student_data (1).csv")


#  Selecting features and target
X = df[["Hours_Studied", "Sleep_Hours", "Attendance"]]
y = df["Score"]


# Splitins data (training & testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  Train models

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# KNN
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)


#  Evaluating the models
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_knn = knn.predict(X_test)

print("\nModel Performance:")
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Decision Tree R2:", r2_score(y_test, y_pred_dt))
print("KNN R2:", r2_score(y_test, y_pred_knn))


#  Preparing new input for prediction
new_data = pd.DataFrame(
    [[hours, sleep, attendance]],
    columns=["Hours_Studied", "Sleep_Hours", "Attendance"]
)


#  Making  predictions
pred_lr = lr.predict(new_data)
pred_dt = dt.predict(new_data)
pred_knn = knn.predict(new_data)

print("\nPredicted Scores:")
print("Linear Regression:", pred_lr[0])
print("Decision Tree:", pred_dt[0])
print("KNN:", pred_knn[0])


#  Visualization
plt.figure(figsize=(8, 6))

# Actual data
plt.scatter(df["Hours_Studied"], df["Score"], label="Actual Data")

# Model predictions
plt.scatter(hours, pred_lr[0], color='red', marker='x', s=100, label="Linear Regression")
plt.scatter(hours, pred_dt[0], color='green', marker='o', s=100, label="Decision Tree")
plt.scatter(hours, pred_knn[0], color='purple', marker='^', s=100, label="KNN")

# Labels and title
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Student Score Prediction (Model Comparison)")

plt.legend()
plt.grid(True)

plt.show()