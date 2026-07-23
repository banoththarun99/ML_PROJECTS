# Bank Customer Churn Prediction Using Machine Learning

## Project Overview

---> Customer churn is one of the biggest challenges faced by banks. Losing existing customers can reduce revenue and increase customer acquisition costs.

---> The goal of this project is to predict whether a customer is likely to leave the bank based on customer information such as age, balance, number of products, salary, and other banking-related features.

---> This project applies multiple machine learning algorithms, compares their performance, and selects the best model for churn prediction.

---

## Problem Statement

Predict whether a customer will leave the bank or continue using banking services.

### Target Variable

* 0 = Customer Stays
* 1 = Customer Leaves (Churn)

---

## Dataset Information

The dataset contains customer demographic and banking information.

### Features Used

* CreditScore
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary
* Satisfaction Score
* Point Earned
* Geography
* Card Type

### Target

* Exited

---

## Project Workflow

### 1. Data Preprocessing

* Loaded the dataset using Pandas
* Removed unnecessary columns
* Removed target leakage feature (`Complain`)
* Prepared feature matrix (X) and target variable (y)

### 2. Train-Test Split

* Training Data: 80%
* Testing Data: 20%

### 3. Machine Learning Models

The following classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier

### 4. Model Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 80.75%   | 58.27%    | 19.85% | 29.62%   |
| Decision Tree       | 78.70%   | 48.05%    | 54.41% | 51.03%   |
| Random Forest       | 85.65%   | 65.55%    | 62.50% | 63.99%   |
| XGBoost             | 86.25%   | 73.33%    | 51.23% | 60.32%   |

---

## Best Model

### Random Forest Classifier

Random Forest was selected as the final model because it achieved:

* High Accuracy
* Strong Precision
* Good Recall
* Highest F1 Score among all evaluated models

### Final Performance

* Accuracy: 85.65%
* Precision: 65.55%
* Recall: 62.50%
* F1 Score: 63.99%

---

## Feature Importance Analysis

The most important features influencing customer churn were:

1. Age
2. NumOfProducts
3. Balance
4. EstimatedSalary
5. Point Earned

These features contributed the most to the model's prediction decisions.

---

## Output Files

The project generates:

* best_model.pkl
* model_comparison.png
* feature_importance.png
* confusion_matrix.png

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* XGBoost
* Joblib

---

## How to Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train_model.py
```

---

## Project Outcome

This project successfully predicts customer churn using machine learning techniques and compares multiple classification algorithms to identify the most effective model. The Random Forest model provided the best balance between precision, recall, and overall performance.

---

## Author

Banoth Tharun

B.Tech – Artificial Intelligence and Data Science
