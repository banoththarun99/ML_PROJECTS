# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Step 3: Split features and target
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Find best K
accuracies = []

print("\nKNN Accuracy for different K values:")

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    accuracies.append(acc)
    print(f"K = {k} → Accuracy = {acc}")

# Best K
best_k = accuracies.index(max(accuracies)) + 1
print("\nBest K:", best_k)

# Train final KNN with best K
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Step 5: Other models

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)

# KNN final accuracy
y_pred_knn = best_knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred_knn)

# Step 6: Model Comparison
print("\nModel Comparison:")
print("Best KNN Accuracy:", knn_acc)
print("Logistic Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)

# Step 7: User Input
print("\nEnter flower measurements:")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

# Convert input
new_sample = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=iris.feature_names
)

# Step 8: Prediction (using best KNN)
prediction = best_knn.predict(new_sample)
predicted_flower = iris.target_names[prediction[0]]

print("\n Predicted Flower:", predicted_flower)

# Store K values
k_values = list(range(1, 11))

lr_line = [lr_acc] * len(k_values)
dt_line = [dt_acc] * len(k_values)

# Plot
plt.figure()

plt.plot(k_values, accuracies, marker='o', label='KNN Accuracy')
plt.plot(k_values, lr_line, linestyle='--', label='Logistic Regression')
plt.plot(k_values, dt_line, linestyle='--', label='Decision Tree')

# Labels
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy Comparison")

plt.legend()
plt.show()