# Credit Scoring Model - Predicting Creditworthiness

## Overview

This project uses Machine Learning techniques to predict whether a customer is creditworthy based on financial attributes such as income, debt level, payment history, loan information, and credit utilization.

Credit scoring is widely used in banking and financial institutions to assess the risk associated with lending money to customers.

---

## Problem Statement

Financial institutions need an efficient way to determine whether a loan applicant is likely to repay borrowed funds.

The objective of this project is to build classification models that predict customer creditworthiness using historical financial indicators.

---

## Dataset

This project uses a synthetic dataset generated using Scikit-Learn's `make_classification()` function.

### Features

* Income
* Debt
* Payment History
* Credit Utilization
* Loan Amount
* Number of Credit Cards
* Number of Loans
* Credit Inquiries

### Target Variable

* `creditworthy`

  * 1 = Creditworthy
  * 0 = Not Creditworthy

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Machine Learning Models

The following classification algorithms were implemented:

### Logistic Regression

A statistical model used for binary classification problems.

### Decision Tree Classifier

A tree-based model that learns decision rules from data.

### Random Forest Classifier

An ensemble learning method that combines multiple decision trees to improve prediction performance.

---

## Project Workflow

1. Generate/Create Dataset
2. Data Preparation
3. Feature Engineering
4. Train-Test Split
5. Feature Scaling using StandardScaler
6. Model Training
7. Performance Evaluation
8. ROC Curve Analysis
9. Model Comparison

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

## Visualization

The project compares model performance using ROC Curves.

ROC curves help evaluate the trade-off between:

* True Positive Rate (TPR)
* False Positive Rate (FPR)

A higher ROC-AUC score indicates better classification performance.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Credit-Scoring-Model
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Run the Project

```bash
python credit_scoring.py
```

---

## Expected Output

The program displays:

* Classification Report
* ROC-AUC Score
* ROC Curve Comparison Plot

for:

* Logistic Regression
* Decision Tree
* Random Forest

---

## Real-World Applications

* Loan Approval Systems
* Banking Risk Assessment
* Credit Card Issuance
* Financial Fraud Prevention
* Consumer Lending Platforms
* FinTech Credit Evaluation Systems

---

## Future Improvements

* Use real-world banking datasets
* Hyperparameter tuning
* Feature selection techniques
* Cross-validation
* Explainable AI (SHAP/LIME)
* Deployment using Streamlit or Flask

---

## Skills Demonstrated

* Data Preprocessing
* Feature Scaling
* Classification Algorithms
* Model Evaluation
* ROC-AUC Analysis
* Data Visualization
* Machine Learning Workflow

---

## Author

Tharun Nayak

Aspiring Data Scientist | Machine Learning Enthusiast | AI Learner
