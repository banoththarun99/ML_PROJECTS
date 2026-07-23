# importing the libraries
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

# LOAD DATA
df = pd.read_csv(
    "data/processed_churn.csv"
)

print("Dataset Loaded Successfully")

# FEATURES AND TARGET
X = df.drop(
    ["Exited", "Complain"],
    axis=1
)

y = df["Exited"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# CREATE MODELS
models = {

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000
    ),


    "Decision Tree":
    DecisionTreeClassifier(
        random_state=42
    ),


    "Random Forest":
    RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),


    "XGBoost":
    XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )
}

# MODEL TRAINING
results = []

best_model = None
best_f1 = 0


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )


    y_pred = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )


    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])


    if f1 > best_f1:
        best_f1 = f1
        best_model = model

# MODEL COMPARISON
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)


print("\n========== MODEL COMPARISON ==========")

print(results_df)

# SAVE BEST MODEL
os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest Model Saved Successfully")
print("Best F1 Score:", best_f1)

# Predictions using best model

y_pred = best_model.predict(X_test)

# Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Confusion Matrix - Best Model"
)

plt.tight_layout()

plt.savefig(
    "output/confusion_matrix.png"
)

plt.show()

print("\nConfusion Matrix Saved Successfully")


# CREATE OUTPUT FOLDER
os.makedirs(
    "output",
    exist_ok=True
)

# MODEL COMPARISON CHART
results_df.set_index("Model").plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Model Performance Comparison")

plt.tight_layout()

plt.savefig(
    "output/model_comparison.png"
)

plt.close()

# CREATE FEATURE IMPORTANCE DF
if hasattr(best_model, "feature_importances_"):
    feature_importances = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_importances.index,
            "Importance": feature_importances.values
        }
    ).sort_values(
        by="Importance",
        ascending=False
    )
else:
    importance_df = pd.DataFrame(
        columns=["Feature", "Importance"]
    )

# Top 10 features
if not importance_df.empty:
    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title("Top Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.tight_layout()

    plt.savefig(
        "output/feature_importance.png"
    )
else:
    print("Feature importance is not available for the selected best model.")

plt.show()